import sys
sys.path.insert(0, ".")
import numpy as np
from batchsim.simulate import simulate_reordered_sum

def test_reordered_sum_delta_non_negative():
    mat = np.ones((8, 8), dtype=np.float32)
    res = simulate_reordered_sum(mat)
    assert res["delta"] >= 0.0
    assert "standard" in res
    assert "reordered" in res
