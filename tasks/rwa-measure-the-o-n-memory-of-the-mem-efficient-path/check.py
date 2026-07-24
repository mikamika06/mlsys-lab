import numpy as np
from mlsys.scorers import size_ratio

def _reference_stats(logits):
    max_per_row = logits.max(axis=1)
    shifted = logits - max_per_row[:, None]
    exp_shifted = np.exp(shifted)
    sum_exp = exp_shifted.sum(axis=1)
    return max_per_row, sum_exp

def grade(sol, fx) -> dict:
    # Use a deterministic test case
    logits = np.array([[0.0, 1.0], [2.0, -1.0]], dtype=np.float64)

    try:
        ref_max, ref_sum = _reference_stats(logits)
        got_max, got_sum = sol.softmax_stats(logits)
    except Exception:
        return {"exact_match": 0.0, "size_ratio": 0.0}

    ok = 1.0 if np.allclose(got_max, ref_max) and np.allclose(got_sum, ref_sum) else 0.0

    # Compute the size ratio using the reference exponential matrix
    max_per_row = logits.max(axis=1)
    shifted = logits - max_per_row[:, None]
    exp_shifted = np.exp(shifted)
    ratio = size_ratio(exp_shifted, ref_max, ref_sum)

    return {"exact_match": ok, "size_ratio": ratio}
