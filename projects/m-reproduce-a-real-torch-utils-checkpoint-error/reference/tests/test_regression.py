import sys
sys.path.insert(0, ".")
from chkpt.optimal import optimal_interval
from chkpt.reproduce import trigger_error
from chkpt.profile import measure


def test_optimal_interval_bounds():
    val = optimal_interval(10, 2.0)
    assert isinstance(val, int)
    assert val >= 1
    assert val <= 10


def test_optimal_interval_zero_layers():
    val = optimal_interval(0, 5.0)
    assert val == 1


def test_reproduce_returns_true():
    assert trigger_error() is True


def test_measure_positive_values():
    m, t = measure(5, True)
    assert m > 0.0
    assert t > 0.0
