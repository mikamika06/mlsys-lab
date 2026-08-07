import sys
sys.path.insert(0, ".")
import numpy as np
from quanteval.eval import compute_relative_error, compute_ir_size_reduction, compute_benchmark_latency_gain


def test_relative_error_bound():
    a = np.ones((4, 4), dtype=np.float32)
    b = np.ones((4, 4), dtype=np.float32) * 1.05
    err = compute_relative_error(a, b)
    assert err < 0.1, f"relative error {err} exceeds bound"


def test_size_reduction_positive():
    ratio = compute_ir_size_reduction(1000, 250)
    assert ratio > 1.0, f"ratio {ratio} should be > 1.0"


def test_latency_gain_positive():
    gain = compute_benchmark_latency_gain(20.0, 10.0)
    assert gain > 1.0, f"gain {gain} should be > 1.0"
