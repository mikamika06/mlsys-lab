import numpy as np
from polyreduce.bisect import bisect_divergent_step
from polyreduce.compare import classify_divergence
from polyreduce.sanitize import sanitize_tensor

def test_bisect_divergence_efficiency_and_accuracy():
    """Regression test for bisection efficiency and correctness."""
    calls = 0
    def check_fn(i):
        nonlocal calls
        calls += 1
        return i < 42

    idx = bisect_divergent_step(100, check_fn)
    assert idx == 42
    assert calls <= 10

def test_classify_fp_noise():
    """Regression test for FP noise classification."""
    a = np.array([1.0, 1e-8, 2.0])
    b = np.array([1.0, 5e-8, 2.0])
    res = classify_divergence(a, b, rtol=1e-9, atol=1e-9, denormal_threshold=1e-7)
    assert res == "FP_NOISE"

def test_sanitize_subnormals():
    """Regression test for subnormal sanitization."""
    a = np.array([1.0, 1e-8, -1e-9, 2.5])
    san = sanitize_tensor(a, denormal_threshold=1e-7)
    assert np.allclose(san, [1.0, 0.0, 0.0, 2.5])
