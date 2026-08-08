import sys
import numpy as np

sys.path.insert(0, ".")
from distill.loss import mse_loss, cosine_loss
from distill.mapping import map_layers, compute_loss_magnitude
from distill.sim import simulate_stability


def test_mse_scale_bounds():
    s = np.ones((2, 4, 8))
    t = np.zeros((2, 4, 8))
    val = mse_loss(s, t)
    assert val < 5.0, f"MSE loss {val} exceeds normal bounds for unit scale"


def test_cosine_loss_range():
    s = np.random.randn(2, 4, 8)
    t = np.random.randn(2, 4, 8)
    val = cosine_loss(s, t)
    assert 0.0 <= val <= 2.0, f"Cosine loss {val} out of bounds [0, 2]"


def test_mapping_coverage():
    s_layers = [0, 1]
    t_layers = [0, 1, 2, 3]
    mapping = map_layers(s_layers, t_layers, "uniform")
    assert len(mapping) == len(s_layers)


def test_stability_detection():
    res = simulate_stability([0.5, 0.6, 12.0, 0.7], threshold=10.0)
    assert not res["stable"]
    assert res["diverge_step"] == 2
