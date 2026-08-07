import sys
import numpy as np

sys.path.insert(0, ".")
from qformat.dequant import dequantize_nvfp4, dequantize_w4a16

def test_nvfp4_global_scale_applied():
    N, K = 4, 32
    gs = 16
    w = np.ones((N, K), dtype=np.float32)
    ls = np.ones((N, K // gs), dtype=np.float32)
    g_scale = 0.5

    out = dequantize_nvfp4(w, ls, g_scale, group_size=gs)
    assert np.allclose(out, 0.5), "Global scale was not correctly applied"

def test_w4a16_ordering():
    packed = np.array([[0x12]], dtype=np.uint8)
    scales = np.array([[1.0]], dtype=np.float32)
    zeros = np.array([[0.0]], dtype=np.float32)

    out = dequantize_w4a16(packed, scales, zeros, group_size=2)
    assert out[0, 0] == 2.0, "Low nibble should map to even index"
    assert out[0, 1] == 1.0, "High nibble should map to odd index"
