import sys
sys.path.insert(0, ".")
from triage.batch import predict_max_batch

def test_batch_limit_positive():
    val = predict_max_batch(16 * 1024 * 1024 * 1024, 2048, 32, 128)
    assert val > 0, "batch limit must be positive"

def test_batch_limit_monotonic():
    v1 = predict_max_batch(16 * 1024 * 1024 * 1024, 1024, 32, 128)
    v2 = predict_max_batch(16 * 1024 * 1024 * 1024, 4096, 32, 128)
    assert v1 >= v2, "larger sequence length should yield smaller or equal max batch size"
