import numpy as np
from evalrep.bootstrap import bootstrap_recovery_ci

def test_bootstrap_bounds():
    base = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    quant = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    res = bootstrap_recovery_ci(base, quant, num_samples=200, alpha=0.05, seed=123)
    assert "lower" in res
    assert "upper" in res
    assert "mean" in res
    assert res["lower"] <= res["mean"] <= res["upper"]
    assert 0.0 <= res["lower"] <= 200.0
