import sys
sys.path.insert(0, ".")
from offload.predict import predict_tok_s

def test_predict_positive():
    val = predict_tok_s(100.0, 1024.0, 0.5)
    assert val > 0.0, f"expected positive throughput, got {val}"

def test_predict_monotonic_bandwidth():
    v1 = predict_tok_s(100.0, 1024.0, 0.5)
    v2 = predict_tok_s(200.0, 1024.0, 0.5)
    assert v2 > v1, f"higher bandwidth should yield higher tok/s: {v1} vs {v2}"
