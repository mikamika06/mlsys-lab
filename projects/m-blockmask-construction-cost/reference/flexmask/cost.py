import math
from typing import Dict


class BlockMaskCostProfiler:
    """Profiler for analyzing BlockMask construction cost and FlexAttention vs FA2 overhead."""

    def __init__(self, block_size: int = 128):
        self.block_size = block_size

    def compute_dense_mask_ops(self, seq_len_q: int, seq_len_k: int) -> int:
        """Compute the total number of point-wise mask evaluations required for a dense mask."""
        return seq_len_q * seq_len_k

    def compute_blockmask_sparse_blocks(
        self, seq_len_q: int, seq_len_k: int, mask_type: str = "causal"
    ) -> Dict[str, int]:
        """Compute block-level metadata counts for blockmask construction."""
        num_blocks_q = math.ceil(seq_len_q / self.block_size)
        num_blocks_k = math.ceil(seq_len_k / self.block_size)
        total_blocks = num_blocks_q * num_blocks_k

        if mask_type == "causal":
            full_blocks = 0
            partial_blocks = 0
            zero_blocks = 0

            for b_q in range(num_blocks_q):
                q_start = b_q * self.block_size
                q_end = min(seq_len_q, (b_q + 1) * self.block_size)

                for b_k in range(num_blocks_k):
                    k_start = b_k * self.block_size
                    k_end = min(seq_len_k, (b_k + 1) * self.block_size)

                    if k_end - 1 <= q_start:
                        full_blocks += 1
                    elif k_start > q_end - 1:
                        zero_blocks += 1
                    else:
                        partial_blocks += 1
        else:
            full_blocks = total_blocks
            partial_blocks = 0
            zero_blocks = 0

        return {
            "total_blocks": total_blocks,
            "full_blocks": full_blocks,
            "partial_blocks": partial_blocks,
            "zero_blocks": zero_blocks,
        }

    def simulate_flex_vs_fa2_latency(
        self,
        seq_len: int,
        num_heads: int,
        blockmask_construction_us_per_block: float = 0.5,
        fa2_kernel_us_per_gflop: float = 2.0,
        flex_kernel_us_per_gflop: float = 2.2,
    ) -> Dict[str, float]:
        """Simulate FlexAttention vs FA2 execution cost components."""
        blocks = self.compute_blockmask_sparse_blocks(seq_len, seq_len, "causal")
        num_grid_blocks = blocks["total_blocks"]

        flex_construction_us = num_grid_blocks * blockmask_construction_us_per_block

        flops = num_heads * (seq_len ** 2)
        gflops = flops / 1e9

        fa2_latency_us = gflops * fa2_kernel_us_per_gflop
        active_ratio = (blocks["full_blocks"] + blocks["partial_blocks"]) / max(1, num_grid_blocks)
        flex_kernel_us = (gflops * active_ratio) * flex_kernel_us_per_gflop

        flex_total_us = flex_construction_us + flex_kernel_us
        latency_ratio = flex_total_us / max(1e-6, fa2_latency_us)

        return {
            "fa2_latency_us": float(fa2_latency_us),
            "flex_construction_us": float(flex_construction_us),
            "flex_kernel_us": float(flex_kernel_us),
            "flex_total_us": float(flex_total_us),
            "latency_ratio": float(latency_ratio),
        }
