import sys

sys.path.insert(0, ".")
import numpy as np
from dfkd.bounds import min_diversity_bound

def test_diversity_bound_accounts_for_truncation():
    teacher_logits = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])
    # K=2 visited: rank=1 SVD leaves 1 singular value of 10.0 unaccounted (error=100)
    # unvisited error (1 row) = 100. Total error = 200. MSE = 200/9 = 22.22
    # If truncation error were ignored, MSE would seem to be 100/9 = 11.11
    k = min_diversity_bound(teacher_logits, rank=1, target_mse=15.0)
    assert k == 3, f"Expected K=3, but got {k} (did it ignore truncation error?)"
