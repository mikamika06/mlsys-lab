import numpy as np
from tf32guard.tolerance import suggest_tolerance
from tf32guard.verify import verify_output


def test_regression_tolerance_scaling():
    shape = (128, 128)
    tol_low = suggest_tolerance(shape, 1.0)
    tol_high = suggest_tolerance(shape, 1000.0)
    assert tol_high > tol_low


def test_regression_verification():
    a = np.ones((64, 64), dtype=np.float32)
    b = a * (1.0 + 1e-5)
    assert verify_output(a, b, 1e-3) is True
    assert verify_output(a, b, 1e-8) is False
