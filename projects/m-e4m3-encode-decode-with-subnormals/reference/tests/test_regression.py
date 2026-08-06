import numpy as np
from fp8util.quant import encode_e4m3, decode_e4m3
from fp8util.scale import compute_scale


def test_subnormal_roundtrip():
    vals = np.array([0.0, 1e-4, 2e-4, 5e-4], dtype=np.float32)
    encoded = encode_e4m3(vals)
    decoded = decode_e4m3(encoded)
    assert np.all(np.isfinite(decoded))


def test_scale_bounds():
    x = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    scale = compute_scale(x)
    assert scale > 0.0
