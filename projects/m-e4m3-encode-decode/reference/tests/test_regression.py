import numpy as np
from fp8.descale import compute_scale
from fp8.e4m3 import E4M3_MAX


def test_e4m3_max_scale():
    x = np.array([100.0, -200.0, 50.0], dtype=np.float32)
    scale = compute_scale(x)
    expected_scale = E4M3_MAX / 200.0
    assert np.isclose(
        scale, expected_scale
    ), f"Scale computation used wrong max float limit: got {scale}, expected {expected_scale}"
