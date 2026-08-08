import sys
import numpy as np

sys.path.insert(0, ".")
from reduction.variance import one_pass_variance, two_pass_variance

def test_variance_stability():
    np.random.seed(42)
    # Generate data with high mean and low variance,
    # perfect for breaking one-pass float16 variance.
    x = np.random.normal(15.0, 0.1, 1000).astype(np.float16)

    true_var = np.var(x.astype(np.float64))

    one = float(one_pass_variance(x))
    two = float(two_pass_variance(x))

    err_one = abs(one - true_var) / true_var
    err_two = abs(two - true_var) / true_var

    assert err_two < 0.1, f"two-pass error is unexpectedly high: {err_two}"
    assert err_one > 0.5, f"one-pass error is unexpectedly low: {err_one}"
