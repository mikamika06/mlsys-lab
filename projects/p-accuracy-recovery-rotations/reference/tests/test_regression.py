import sys
sys.path.insert(0, ".")
import numpy as np
from quant_rec.analysis import measure_layer_error
from quant_rec.rotations import apply_rotation
from quant_rec.rounding import optimize_rounding
from quant_rec.lowrank import apply_low_rank_corrector

def test_measure_error_basic():
    w = np.array([[1.0, 2.0], [3.0, 4.0]])
    qw = np.array([[1.1, 1.9], [3.1, 3.9]])
    err = measure_layer_error(w, qw)
    assert err > 0.0

def test_rotation_reduces_outliers():
    w = np.array([[10.0, 0.1], [0.1, -10.0]])
    mat = np.array([[0.707, 0.707], [-0.707, 0.707]])
    _, m = apply_rotation(w, mat)
    assert m < 15.0

def test_rounding_grid():
    w = np.array([0.1, 0.4, 0.8])
    grid = np.array([0.0, 0.5, 1.0])
    res = optimize_rounding(w, grid)
    assert res.shape == w.shape

def test_low_rank_cost():
    res = np.random.randn(8, 8)
    _, cost = apply_low_rank_corrector(res, 2)
    assert cost > 0.0
