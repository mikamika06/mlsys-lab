import numpy as np
from fp8codec.optimize import optimize_scale

def test_optimal_scale():
    x = np.array([-10.0, 0.0, 20.0, 44.8], dtype=np.float32)
    scale = optimize_scale(x)
    assert scale > 0.0
    assert np.isfinite(scale)
    reconstructed_scale = x / scale
    assert np.all(np.abs(reconstructed_scale) <= 448.0 + 1e-5)
