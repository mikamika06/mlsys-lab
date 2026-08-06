import numpy as np
from quantizer.conv import integer_conv2d

def test_zero_point_invariant():
    i_q = np.array([[[[10]]]], dtype=np.uint8)
    i_z = 10
    w_q = np.array([[[[1]]]], dtype=np.int8)
    b_q = np.array([0], dtype=np.int32)
    
    out = integer_conv2d(i_q, i_z, w_q, b_q)
    assert out[0, 0, 0, 0] == 0, f"Expected 0 accumulator due to zero point, got {out[0, 0, 0, 0]}"
