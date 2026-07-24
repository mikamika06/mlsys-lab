import numpy as np
from mlsys.scorers import rel_err

def _reference(up_proj, down_proj):
    # Compute group L2 norm per hidden neuron
    row_norm_sq = np.sum(up_proj**2, axis=1)
    col_norm_sq = np.sum(down_proj**2, axis=0)
    return np.sqrt(row_norm_sq + col_norm_sq)

def grade(sol, fx) -> dict:
    # Generate a few random test cases
    rng = np.random.default_rng(42)
    cases = [
        (rng.standard_normal((5, 3)), rng.standard_normal((4, 5))),
        (rng.standard_normal((10, 8)), rng.standard_normal((6, 10))),
        (rng.standard_normal((7, 1)), rng.standard_normal((2, 7))),
        (np.zeros((3, 0)), np.zeros((0, 3))),   # edge case: zero input dims
    ]
    max_err = 0.0
    for up, down in cases:
        try:
            got = sol.per_neuron_importance(up, down)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _reference(up, down)
        # Ensure shape and dtype match
        if got.shape != ref.shape or got.dtype != np.float64:
            return {"rel_err": float("inf")}
        err = rel_err(ref, got)
        max_err = max(max_err, err)
    return {"rel_err": max_err}
