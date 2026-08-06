import numpy as np
import pytest
from prune.wanda import wanda_prune


def test_calibration_scaling():
    np.random.seed(42)
    W = np.random.randn(16, 32)
    X = np.random.randn(100, 32)

    res1 = wanda_prune(W, X, 0.5, domain_shift=False)
    res2 = wanda_prune(W, X * 2.0, 0.5, domain_shift=False)

    assert np.allclose(res1["pruned_weights"], res2["pruned_weights"], atol=1e-5)
