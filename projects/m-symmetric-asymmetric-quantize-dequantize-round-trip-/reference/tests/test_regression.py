import numpy as np
from quant.numerics import calc_scale_zp_asymmetric, dequantize_asymmetric, quantize_asymmetric


def test_zero_point_invariant():
    scale, zp = calc_scale_zp_asymmetric(-5.0, 5.0, bits=8)
    assert zp != 0

    x = np.array([0.0])
    q = quantize_asymmetric(x, scale, zp, bits=8)
    assert q[0] == zp

    dq = dequantize_asymmetric(q, scale, zp)
    assert np.isclose(dq[0], 0.0)
