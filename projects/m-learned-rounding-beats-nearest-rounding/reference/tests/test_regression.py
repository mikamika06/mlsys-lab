import sys
import numpy as np

sys.path.insert(0, ".")
from quantlib.rounding import nearest_rounding, learned_rounding
from quantlib.autoround import AutoRoundModifier


def test_learned_rounding_beats_nearest():
    np.random.seed(42)
    w = np.random.randn(16, 16) * 0.5
    scale = 0.1
    zp = 0.0
    nn_res = nearest_rounding(w, scale, zp)
    lr_res = learned_rounding(w, scale, zp, steps=10)
    mse_nn = np.mean((w - nn_res) ** 2)
    mse_lr = np.mean((w - lr_res) ** 2)
    assert mse_lr <= mse_nn, f"Learned MSE {mse_lr} not better than nearest MSE {mse_nn}"


def test_autoround_modifier_runs():
    model = {"weights": [np.random.randn(8, 8)], "scale": 0.1, "zero_point": 0.0}
    modifier = AutoRoundModifier(model)
    res = modifier.optimize([np.zeros((8, 8))])
    assert len(res) == 1
    assert res[0].shape == (8, 8)
