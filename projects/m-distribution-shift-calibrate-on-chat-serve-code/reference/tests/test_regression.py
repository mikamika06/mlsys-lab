import sys
sys.path.insert(0, ".")
import numpy as np
from calib.shift import compute_shift
from calib.metrics import relative_error
from calib.adjust import adjust_scales


def test_shift_computation_shape():
    chat = np.ones((10, 16))
    code = np.ones((10, 16)) * 2.0
    shift = compute_shift(chat, code)
    assert shift.shape == (16,)


def test_relative_error_bound():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.1, 1.9, 3.1])
    err = relative_error(a, b)
    assert err < 0.2


def test_adjust_scales_limits():
    scales = np.array([1.0, 1.0])
    shift = np.array([0.1, 5.0])
    adjusted = adjust_scales(scales, shift)
    assert adjusted[0] == 0.5
    assert adjusted[1] == 2.0
