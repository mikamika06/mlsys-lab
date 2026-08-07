import sys
sys.path.insert(0, ".")
from roofline.model import compute_decode_roofline
from roofline.sweep import validate_sweep

def test_roofline_positive_throughput():
    cfg = {"hidden_size": 4096, "num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "vocab_size": 32000, "weight_bytes": 14 * 10**9}
    res = compute_decode_roofline(cfg, 4, 900.0 * 10**9, 300.0 * 10**12)
    assert res > 0.0

def test_validate_sweep_bounds():
    cfg = {"hidden_size": 4096, "num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "vocab_size": 32000, "weight_bytes": 14 * 10**9}
    sw = {
        "batch_sizes": [1, 2, 4],
        "measured_tokens_per_sec": [1200.0, 2300.0, 4400.0],
        "memory_bandwidth": 900.0 * 10**9,
        "compute_capacity": 300.0 * 10**12
    }
    out = validate_sweep(sw, cfg, max_rel_err=0.50)
    assert 0.0 <= out["max_rel_err"] <= 1.0
    assert "passed" in out
