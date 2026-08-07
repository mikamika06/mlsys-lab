import sys
import numpy as np

sys.path.insert(0, ".")
from attencurve.rescale import rescale_block

def test_rescale_correctness():
    rm, rs = 1.0, 2.0
    bm, bs = 3.0, 4.0
    nm, ns = rescale_block(rm, rs, bm, bs, use_old_max=False)
    assert nm == 3.0
    expected_ns = rs * np.exp(1.0 - 3.0) + bs * np.exp(3.0 - 3.0)
    assert np.isclose(ns, expected_ns)

def test_rescale_old_max_behavior():
    rm, rs = 2.0, 1.0
    bm, bs = 1.0, 3.0
    nm, ns = rescale_block(rm, rs, bm, bs, use_old_max=True)
    assert nm == 2.0
    expected_ns = rs + bs * np.exp(1.0 - 2.0)
    assert np.isclose(ns, expected_ns)
