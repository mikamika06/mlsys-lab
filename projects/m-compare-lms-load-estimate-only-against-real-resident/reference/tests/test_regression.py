import sys

sys.path.insert(0, ".")
from lmsmem import estimate_bytes, resident_bytes, relative_error, offload_split

CONFIGS = [
    {"n_layers": 6, "hidden_size": 512, "n_heads": 8, "n_kv_heads": 8, "head_dim": 64,
     "intermediate_size": 1376, "vocab_size": 32000, "tie_embeddings": True},
    {"n_layers": 16, "hidden_size": 2048, "n_heads": 16, "n_kv_heads": 4, "head_dim": 128,
     "intermediate_size": 5632, "vocab_size": 32000, "tie_embeddings": False},
]


def test_resident_always_exceeds_estimate():
    for cfg in CONFIGS:
        for ctx in (2048, 8192):
            est = estimate_bytes(cfg, ctx, 2.0)
            res = resident_bytes(cfg, ctx, 2.0)
            assert res > est, f"resident {res} did not exceed estimate {est}"


def test_relative_error_is_positive():
    est = estimate_bytes(CONFIGS[0], 4096, 2.0)
    res = resident_bytes(CONFIGS[0], 4096, 2.0)
    assert relative_error(est, res) > 0.0, "relative_error must be > 0 when resident exceeds estimate"


def test_offload_split_sums_to_total():
    total = resident_bytes(CONFIGS[1], 8192, 2.0)
    for ratio in (0.0, 0.3, 0.7, 1.0):
        split = offload_split(total, ratio)
        assert split["gpu_bytes"] + split["cpu_bytes"] == total, f"split {split} does not sum to {total}"
