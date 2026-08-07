import numpy as np
from calib.scales import compute_max_scale
from calib.entropy import compute_entropy_scale
from calib.nvfp4 import nvfp4_round_trip


def test_max_scale_positive():
    x = np.random.randn(32, 32).astype(np.float32)
    scale = compute_max_scale(x)
    assert scale > 0.0


def test_entropy_scale_positive():
    x = np.random.randn(32, 32).astype(np.float32)
    scale = compute_entropy_scale(x)
    assert scale > 0.0


def test_nvfp4_round_trip_shape():
    x = np.random.randn(16, 16).astype(np.float32)
    out = nvfp4_round_trip(x, block_size=16)
    assert out.shape == x.shape


def test_nvfp4_round_trip_closeness():
    x = np.random.randn(16, 16).astype(np.float32) * 0.1
    out = nvfp4_round_trip(x, block_size=16)
    assert np.all(np.abs(out - x) < 1.0)
