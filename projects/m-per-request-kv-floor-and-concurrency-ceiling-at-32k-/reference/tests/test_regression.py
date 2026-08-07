import sys
sys.path.insert(0, ".")
from kvcapacity.feasibility import concurrency_ceiling

MODEL_CONFIG = {
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_size": 4096,
    "max_position_embeddings": 131072,
    "num_parameters": 20000000000,
}

def test_tp_scales_kv_cache_capacity():
    c1 = concurrency_ceiling(80.0, MODEL_CONFIG, tp_size=1, model_dtype="float16", kv_dtype="float16", seq_len=131072)
    c4 = concurrency_ceiling(80.0, MODEL_CONFIG, tp_size=4, model_dtype="float16", kv_dtype="float16", seq_len=131072)
    assert c4 >= 10, f"Expected TP=4 capacity >= 10, got {c4}"
    assert c4 > c1, f"Expected TP=4 concurrency ({c4}) > TP=1 ({c1})"

def test_invalid_tp_head_division_returns_zero():
    cfg = dict(MODEL_CONFIG, num_key_value_heads=6)
    c4 = concurrency_ceiling(80.0, cfg, tp_size=4, model_dtype="float16", kv_dtype="float16", seq_len=131072)
    assert c4 == 0, f"Expected concurrency 0 for non-divisible KV heads, got {c4}"
