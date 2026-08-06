import numpy as np
from quantlibs.dequant import dequantize_to_fp16

def test_dequantization_bounds():
    data = [120, 130, 140]
    scale = 0.1
    zero_point = 128
    res = dequantize_to_fp16(data, scale, zero_point)
    expected = np.array([(120 - 128) * 0.1, (130 - 128) * 0.1, (140 - 128) * 0.1], dtype=np.float16)
    np.testing.assert_allclose(res, expected, rtol=1e-5, atol=1e-5)
