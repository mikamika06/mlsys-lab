import sys
sys.path.insert(0, ".")
from qcache.error import compute_quant_error
import numpy as np

def test_quant_error_ordering():
    np.random.seed(0)
    t = np.random.randn(10, 10)
    e4 = compute_quant_error(t, 4)
    e2 = compute_quant_error(t, 2)
    assert e2 > e4, "2-bit error should be greater than 4-bit error"
    assert e4 >= 0.0
