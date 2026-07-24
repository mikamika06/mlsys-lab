import numpy as np
from mlsys.scorers import max_abs_err

def _ref(head_slopes: np.ndarray, seq_len: int) -> np.ndarray:
    """Reference implementation used by the grader."""
    head_slopes = np.asarray(head_slopes, dtype=np.float64)
    pos = np.arange(seq_len, dtype=np.int32)
    dist = pos[:, None] - pos[None, :]  # shape (L,L)
    biases = -head_slopes[:, None, None] * dist[None, :, :]
    return biases

def grade(sol, fx) -> dict:
    cases = [
        (np.array([0.01, 0.02]), 4),
        (np.array([0.05]), 3),
        (np.array([0.1, 0.2, 0.3]), 5)
    ]
    max_err = 0.0
    for slopes, L in cases:
        try:
            got = sol.alibi_bias_matrix(slopes, L)
        except Exception:
            return {"max_abs_err": float("inf")}
        expected = _ref(slopes, L)
        err = max_abs_err(expected, got)
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
