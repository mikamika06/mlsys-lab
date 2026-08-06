import sys
sys.path.insert(0, ".")
from kvcalc.memory import compute_kv_bytes
from kvcalc.budget import max_context_length
from kvcalc.oom import back_calculate_oom


def test_compute_kv_bytes_scaling():
    cfg = {
        "name": "test-mha",
        "type": "mha",
        "num_layers": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "bytes_per_elem": 2,
    }
    b1 = compute_kv_bytes(cfg, 1000, 1)
    b2 = compute_kv_bytes(cfg, 2000, 1)
    assert b2 == 2 * b1


def test_max_context_length_positive():
    cfg = {
        "name": "test-mha",
        "type": "mha",
        "num_layers": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "bytes_per_elem": 2,
    }
    ml = max_context_length(cfg, 36 * 1024 * 1024 * 1024, 1)
    assert ml > 0


def test_back_calculate_oom_correctness():
    cfg = {
        "name": "test-mha",
        "type": "mha",
        "num_layers": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "bytes_per_elem": 2,
    }
    max_len = max_context_length(cfg, 36 * 1024 * 1024 * 1024, 1)
    failed_len = max_len + 5000
    safe_len = back_calculate_oom(cfg, failed_len, 1)
    assert safe_len == max_len
