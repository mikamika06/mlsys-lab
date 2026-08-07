import sys
sys.path.insert(0, ".")
from serve.metrics import compute_throughput_ratio, compute_memory_delta


def test_throughput_ratio_positive():
    r = compute_throughput_ratio(100.0, 135.0)
    assert r > 0.0


def test_memory_delta_non_negative():
    d = compute_memory_delta(1000, 400)
    assert d >= 0.0
