import sys
sys.path.insert(0, ".")
from hybridkv.config import classify_attention
from hybridkv.memory import effective_bytes_per_token
from hybridkv.predict import predict_startup_kv_size

CONFIG = {
    "layers": [
        {"index": 0, "kind": "full", "kv_heads": 4, "head_dim": 64},
        {"index": 1, "kind": "sliding", "window": 128, "kv_heads": 4, "head_dim": 64},
        {"index": 2, "kind": "full", "kv_heads": 4, "head_dim": 64}
    ]
}

def test_classification_correct():
    assert classify_attention(CONFIG["layers"][0]) == "full"
    assert classify_attention(CONFIG["layers"][1]) == "sliding"

def test_effective_bytes_bounded_by_window():
    b_small = effective_bytes_per_token(CONFIG, 64)
    b_large = effective_bytes_per_token(CONFIG, 512)
    assert b_small <= b_large

def test_prediction_matches_formula():
    seq = 256
    pred = predict_startup_kv_size(CONFIG, seq)
    actual = effective_bytes_per_token(CONFIG, seq)
    assert pred == actual
