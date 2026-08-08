import numpy as np
import sys
sys.path.insert(0, ".")
from quant_observe.observer import minmax_observer, mse_observer
from quant_observe.bias import ignored_zp_bias

def test_mse_never_worse_than_minmax():
    x = np.random.randn(200).astype(np.float32) * 10
    x[0] = 500.0
    args = {"bits": 8, "symmetric": True}

    s_min, z_min = minmax_observer(x, args)
    s_mse, z_mse = mse_observer(x, args)

    def get_mse(s, z):
        xq = np.clip(np.round(x / s) + z, -128, 127)
        return np.mean((x - (xq - z) * s)**2)

    assert get_mse(s_mse, z_mse) <= get_mse(s_min, z_min) + 1e-5

def test_ignored_zp_bias_is_linear():
    x = np.random.rand(150).astype(np.float32) * 5
    args = {"bits": 4, "symmetric": False}

    b = ignored_zp_bias(x, args, "minmax")
    s, z = minmax_observer(x, args)

    expected_bias = s * z * len(x)
    assert abs(b - expected_bias) < 1e-4
