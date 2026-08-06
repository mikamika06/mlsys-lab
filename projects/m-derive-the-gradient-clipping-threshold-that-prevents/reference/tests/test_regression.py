import sys
sys.path.insert(0, ".")
from numerics.underflow import apply_grad_scaler
from numerics.threshold import compute_clip_threshold
import numpy as np

def test_grad_scaler_preserves_magnitude():
    small_grad = 1e-7
    res = apply_grad_scaler(small_grad, scale_factor=10000.0)
    assert res != 0.0

def test_threshold_positive():
    grads = [np.array([1.0, 2.0])]
    t = compute_clip_threshold(grads, 1.0)
    assert t > 0.0
