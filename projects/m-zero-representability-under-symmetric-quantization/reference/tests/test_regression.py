import sys
import numpy as np

sys.path.insert(0, ".")
from quant.sym import compute_scale, quantize, dequantize


def test_zero_maps_to_exact_zero():
    t = np.array([0.0, -1.2, 3.4], dtype=np.float32)
    scale = compute_scale(t, -128, 127)
    codes = quantize(t, scale, -128, 127)
    deq = dequantize(codes, scale)
    assert deq[0] == 0.0
    assert codes[0] == 0


def test_symmetric_scale_positive():
    t = np.array([-5.0, 5.0], dtype=np.float32)
    scale = compute_scale(t, -128, 127)
    assert scale > 0.0
