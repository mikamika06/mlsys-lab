import numpy as np
from dtypecheck.reduction import safe_reduction

def test_safe_reduction_overflow():
    t = np.array([70000.0, 80000.0], dtype=np.float32)
    res = safe_reduction(t)
    assert not np.isnan(res)
    assert not np.isinf(res)
    assert res == 150000.0
