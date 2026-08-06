import sys
sys.path.insert(0, ".")
from kvquant.memory import calculate_kv_cache_bytes, evaluate_perplexity_delta


def test_kv_bytes_scaling():
    b16 = calculate_kv_cache_bytes(32, 8, 128, 32768, "f16")
    b32 = calculate_kv_cache_bytes(32, 8, 128, 65536, "f16")
    assert b32 == 2 * b16, "KV cache bytes should scale linearly with sequence length"


def test_q8_0_cheaper_than_f16():
    f16_bytes = calculate_kv_cache_bytes(32, 8, 128, 32768, "f16")
    q8_bytes = calculate_kv_cache_bytes(32, 8, 128, 32768, "q8_0")
    assert q8_bytes < f16_bytes
    assert q8_bytes == int(f16_bytes * (34 / 64))


def test_q4_0_cheaper_than_q8_0():
    q8_bytes = calculate_kv_cache_bytes(32, 8, 128, 32768, "q8_0")
    q4_bytes = calculate_kv_cache_bytes(32, 8, 128, 32768, "q4_0")
    assert q4_bytes < q8_bytes
    assert q4_bytes == int(q8_bytes * (18 / 34))


def test_perplexity_delta_order():
    d_f16 = evaluate_perplexity_delta(6.5, "f16", 32768)
    d_q8 = evaluate_perplexity_delta(6.5, "q8_0", 32768)
    d_q4 = evaluate_perplexity_delta(6.5, "q4_0", 32768)
    assert d_f16 == 0.0
    assert 0.0 < d_q8 < d_q4
