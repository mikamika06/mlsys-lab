import sys
import numpy as np

sys.path.insert(0, ".")
from system import distributed, analysis

def test_reduction_is_deterministic():
    tensors = [np.array([10000.0], dtype=np.float32)]
    for _ in range(30):
        tensors.append(np.array([1.0], dtype=np.float32))

    diff = analysis.check_determinism(distributed.safe_all_reduce_sum, tensors)
    assert diff == 0.0, f"Reduction is not deterministic, diff is {diff}"
