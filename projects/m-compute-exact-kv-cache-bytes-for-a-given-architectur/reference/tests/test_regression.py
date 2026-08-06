from kvcalc.calc import compute_kv_cache_bytes
from kvcalc.gpu import find_max_gpu_context


def test_compute_kv_bytes_exact_value():
    cfg = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2}
    res = compute_kv_cache_bytes(cfg, 2048)
    expected = 2 * 32 * 8 * 128 * 2 * 2048
    assert res == expected


def test_find_max_gpu_context_fixed_vram():
    cfg = {"num_layers": 32, "num_kv_heads": 8, "head_dim": 128, "bytes_per_elem": 2}
    weights = 0
    per_token = 2 * 32 * 8 * 128 * 2
    vram = per_token * 1000 + 500
    assert find_max_gpu_context(cfg, weights, vram) == 1000
