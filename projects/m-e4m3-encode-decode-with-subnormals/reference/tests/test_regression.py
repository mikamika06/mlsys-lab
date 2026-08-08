import sys
sys.path.insert(0, ".")
import numpy as np
from quantlib.codec import encode_e4m3, decode_e4m3
from quantlib.scale import compute_scale
from quantlib.compare import compare_formats


def test_codec_roundtrip():
    x = np.array([0.0, 1.0, -1.0, 0.05, -0.05], dtype=np.float32)
    encoded = encode_e4m3(x)
    decoded = decode_e4m3(encoded)
    assert decoded.shape == x.shape


def test_scale_positive():
    x = np.array([-2.5, 0.0, 3.1], dtype=np.float32)
    s = compute_scale(x)
    assert s > 0.0


def test_compare_ordering():
    x = np.array([0.1, 0.5, 2.0], dtype=np.float32)
    res = compare_formats(x)
    assert "e4m3_mse" in res
    assert res["e4m3_mse"] >= 0.0
