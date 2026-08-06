import sys
sys.path.insert(0, ".")
import numpy as np
from quant.scheme import parse_scheme
from quant.observer import MinMaxObserver, MSEObserver
from quant.bias import compute_zero_point_bias


def test_scheme_parsing():
    args = parse_scheme("int8-sym-tensor")
    assert args.bits == 8
    assert args.symmetric is True
    assert args.granularity == "tensor"


def test_minmax_observer():
    obs = MinMaxObserver(bits=8, symmetric=True)
    obs.update(np.array([-10.0, 5.0, 2.0]))
    scale, zp = obs.compute_params()
    assert scale > 0
    assert zp == 0


def test_mse_observer_bounds():
    obs = MSEObserver(bits=8, symmetric=True)
    obs.update(np.array([-5.0, -1.0, 2.0, 4.0]))
    scale, zp = obs.compute_params()
    assert scale > 0


def test_bias_computation():
    x = np.array([1.0, 2.0, 3.0])
    bias = compute_zero_point_bias(x, scale=0.1, zero_point=10)
    assert isinstance(bias, float)
