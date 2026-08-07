import sys
sys.path.insert(0, ".")
from prefixcache.reuse import find_reusable_blocks
from prefixcache.hybrid import compute_mamba_state_size
from prefixcache.cost import compute_dense_vs_hybrid_cost

CONFIG = {
    "total_tokens": 16384,
    "block_size": 16,
    "layers": [
        {"index": 0, "kind": "full", "kv_heads": 8, "head_dim": 128},
        {"index": 1, "kind": "sliding", "window": 1024, "kv_heads": 8, "head_dim": 128},
        {"index": 2, "kind": "mamba", "state_dim": 64, "d_inner": 2048},
    ]
}


def test_reusable_blocks_not_empty():
    blocks = find_reusable_blocks(CONFIG)
    assert len(blocks) > 0, "expected some reusable blocks within sliding window bounds"


def test_mamba_state_size_positive():
    layer = CONFIG["layers"][2]
    size = compute_mamba_state_size(layer)
    assert size > 0, "mamba state size must be positive"


def test_hybrid_cost_less_than_dense():
    costs = compute_dense_vs_hybrid_cost(CONFIG, 131072)
    assert costs["hybrid_bytes"] < costs["dense_bytes"], "hybrid model should cost less memory than dense at 128k context"
