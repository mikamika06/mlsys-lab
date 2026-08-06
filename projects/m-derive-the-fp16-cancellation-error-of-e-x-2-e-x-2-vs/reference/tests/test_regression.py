import numpy as np
from variance.derivation import compute_fp16_variance

def test_variance_error_bounds():
    data = np.ones((10, 32), dtype=np.float32) * 50.0
    err = compute_fp16_variance(data)
    assert err >= 0.0
    assert err < 10.0
