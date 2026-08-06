import numpy as np
from quantfix.hessian import validate_hessian
from quantfix.decision import should_use_model_free_ptq


def test_validate_hessian_nan():
    h_bad = np.eye(3, dtype=np.float32)
    h_bad[0, 0] = np.nan
    assert validate_hessian(h_bad) is False


def test_validate_hessian_valid():
    h_good = np.eye(3, dtype=np.float32)
    assert validate_hessian(h_good) is True


def test_decision_model_free_on_nan():
    h_bad = np.eye(3, dtype=np.float32)
    h_bad[1, 1] = np.inf
    assert should_use_model_free_ptq(h_bad) is True
