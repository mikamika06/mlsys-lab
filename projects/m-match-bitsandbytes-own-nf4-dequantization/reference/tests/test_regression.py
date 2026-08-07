import sys
import numpy as np

sys.path.insert(0, ".")
from nf4.dequant import dequantize_nf4, unpack_4bit


def test_unpack_order():
    packed = np.array([0x12], dtype=np.uint8)
    unpacked = unpack_4bit(packed)
    assert unpacked[0] == 2
    assert unpacked[1] == 1


def test_dequantize_scaling():
    packed = np.array([0x7F], dtype=np.uint8)
    absmax = np.array([2.0], dtype=np.float32)
    out = dequantize_nf4(packed, absmax, blocksize=2)
    assert np.abs(out[0] - 2.0) < 1e-6
    assert np.abs(out[1] - 0.0) < 1e-6
