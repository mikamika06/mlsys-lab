import sys
import numpy as np

sys.path.insert(0, ".")
from quant.core import calc_qparams

def test_asymmetric_respects_bounds_and_nonzero_min():
    w_view = np.array([[-50.0, 205.0]])
    scale, zp = calc_qparams(w_view, symmetric=False)

    assert np.isclose(scale[0, 0], 1.0), f"Expected scale 1.0 for span 255, got {scale[0, 0]}"
    assert zp[0, 0] == 50.0, f"Expected zero point 50, got {zp[0, 0]}"
