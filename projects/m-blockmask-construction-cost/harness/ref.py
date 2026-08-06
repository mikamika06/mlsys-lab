from flexmask.cost import BlockMaskCostProfiler
from flexmask.cache import MaskCache, DummyBlockMask

TEST_CONFIGS = [
    {"seq_len": 2048, "heads": 16, "block_size": 128},
    {"seq_len": 4096, "heads": 32, "block_size": 128},
    {"seq_len": 8192, "heads": 32, "block_size": 64},
]


def run_profiler_reference(cfg):
    profiler = BlockMaskCostProfiler(block_size=cfg["block_size"])
    dense_ops = profiler.compute_dense_mask_ops(cfg["seq_len"], cfg["seq_len"])
    blocks = profiler.compute_blockmask_sparse_blocks(
        cfg["seq_len"], cfg["seq_len"], "causal"
    )
    sim = profiler.simulate_flex_vs_fa2_latency(
        seq_len=cfg["seq_len"], num_heads=cfg["heads"]
    )
    return {"dense_ops": dense_ops, "blocks": blocks, "sim": sim}
