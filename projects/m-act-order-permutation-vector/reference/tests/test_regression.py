import numpy as np
from gptq.core import act_order_perm

def test_act_order_direction():
    H = np.diag([1.0, 5.0, 3.0])
    perm = act_order_perm(H)
    assert list(perm) == [1, 2, 0], "Must sort descending by diagonal"
