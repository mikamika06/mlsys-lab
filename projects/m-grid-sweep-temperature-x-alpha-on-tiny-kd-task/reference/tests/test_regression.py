import numpy as np
from kd_sweep.sweep import run_sweep
from kd_sweep.gradient import verify_gradient_scaling
from kd_sweep.noise import effective_temperature

def test_sweep_shape():
    tl = np.zeros((1, 3))
    sl = np.zeros((1, 3))
    lbl = np.array([[1.0, 0.0, 0.0]])
    res = run_sweep(tl, sl, lbl, [1.0, 2.0], [0.1, 0.9])
    assert res.shape == (2, 2)

def test_gradient_positive():
    tl = np.array([[1.0, 2.0, 0.5]])
    sl = np.array([[0.5, 1.5, 1.0]])
    val = verify_gradient_scaling(tl, sl, 2.0)
    assert val >= 0.0

def test_noise_shift():
    tl = np.array([[1.0, 2.0, 0.5]])
    val = effective_temperature(tl, 0.1, 2.0)
    assert val > 0.0
