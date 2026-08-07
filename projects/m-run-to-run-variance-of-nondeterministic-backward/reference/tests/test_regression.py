import sys
sys.path.insert(0, ".")
from flashvar.variance import compute_variance
from flashvar.reducer import deterministic_backward


def test_variance_bounds():
    runs = [[1.0, 2.0], [1.0001, 2.0001], [0.9999, 1.9999]]
    res = compute_variance(runs)
    assert res["max_rel_err"] < 0.01


def test_deterministic_reducer():
    grads = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    res = deterministic_backward(grads)
    assert len(res) == 2
    assert res[0] == 9.0
    assert res[1] == 12.0
