import sys

sys.path.insert(0, ".")
from gqa_opt.memory import analyze_gpu_expansion_overhead, calculate_kv_cache_bytes

CONFIG_GQA = {
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_layers": 32,
}


def test_native_vs_expanded_memory():
    res = calculate_kv_cache_bytes(CONFIG_GQA, batch_size=4, seq_len=4096)
    assert res["native_bytes"] < res["mha_bytes"]
    assert res["bytes_saved"] == res["mha_bytes"] - res["native_bytes"]


def test_gpu_expansion_overhead_factor():
    overhead = analyze_gpu_expansion_overhead(CONFIG_GQA, batch_size=2, seq_len=2048)
    expected_factor = 32 / 8
    assert abs(overhead["expansion_factor"] - expected_factor) < 1e-6
    assert overhead["expansion_overhead_bytes"] > 0
