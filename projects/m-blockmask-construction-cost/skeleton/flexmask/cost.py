import numpy as np


class BlockMaskCostProfiler:
    """Profiler for analyzing BlockMask construction cost and FlexAttention vs FA2 overhead."""

    def __init__(self, block_size: int = 128):
        self.block_size = block_size

    def compute_dense_mask_ops(self, seq_len_q: int, seq_len_k: int) -> int:
        """Compute the total number of point-wise mask evaluations required for a dense mask."""
        raise NotImplementedError

    def compute_blockmask_sparse_blocks(
        self, seq_len_q: int, seq_len_k: int, mask_type: str = "causal"
    ) -> dict:
        """Compute the count of active, unmasked, and partially masked blocks.

        Returns a dict with keys: 'total_blocks', 'full_blocks', 'partial_blocks', 'zero_blocks'.
        """
        raise NotImplementedError

    def simulate_flex_vs_fa2_latency(
        self,
        seq_len: int,
        num_heads: int,
        blockmask_construction_us_per_block: float = 0.5,
        fa2_kernel_us_per_gflop: float = 2.0,
        flex_kernel_us_per_gflop: float = 2.2,
    ) -> dict:
        """Simulate latency components (construction time, kernel time, total time) for FlexAttention and FA2.

        Returns dict with keys: 'fa2_latency_us', 'flex_construction_us', 'flex_kernel_us',
        'flex_total_us', 'latency_ratio'.
        """
        raise NotImplementedError
