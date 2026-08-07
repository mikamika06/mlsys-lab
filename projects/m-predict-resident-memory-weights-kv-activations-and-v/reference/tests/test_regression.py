import sys

sys.path.insert(0, ".")
from memrunner.bandwidth import predict_decode_tok_s
from memrunner.kquants import explain_kquant_precision_mix
from memrunner.predictor import (
    calculate_activation_bytes,
    calculate_kv_cache_bytes,
    calculate_weight_bytes,
    predict_resident_vram,
)

CONFIG = {
    "num_layers": 32,
    "hidden_dim": 4096,
    "intermediate_dim": 11008,
    "num_heads": 32,
    "num_kv_heads": 8,
    "vocab_size": 32000,
    "quant_type": "Q4_K",
    "kv_bytes_per_elem": 2,
    "act_bytes_per_elem": 2,
}


def test_weight_bytes_differs_from_uniform():
    w_bytes = calculate_weight_bytes(CONFIG)
    total_params = 32 * (4096 * 4096 + 2 * 4096 * 1024 + 4096 * 4096 + 2 * 4096 * 11008 + 11008 * 4096 + 2 * 4096) + 2 * 32000 * 4096
    uniform_4bit_bytes = (total_params * 4.0) / 8.0
    assert w_bytes != uniform_4bit_bytes, "Weight bytes should account for non-uniform K-quant bitwidths"


def test_vram_components_additive():
    w = calculate_weight_bytes(CONFIG)
    kv = calculate_kv_cache_bytes(CONFIG, seq_len=2048, batch_size=1)
    act = calculate_activation_bytes(CONFIG, seq_len=2048, batch_size=1)
    tot = predict_resident_vram(CONFIG, seq_len=2048, batch_size=1)
    assert tot == w + kv + act


def test_kquant_precision_mix_detected():
    exp = explain_kquant_precision_mix(CONFIG)
    assert exp["is_mixed_precision"] is True
    assert exp["attn_bits"] > exp["ffn_down_bits"]


def test_bandwidth_scaling():
    t1 = predict_decode_tok_s(CONFIG, seq_len=1024, memory_bandwidth_gbps=800, batch_size=1)
    t2 = predict_decode_tok_s(CONFIG, seq_len=1024, memory_bandwidth_gbps=1600, batch_size=1)
    assert abs(t2 - 2 * t1) < 1e-4
