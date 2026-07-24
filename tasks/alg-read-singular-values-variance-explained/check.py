import numpy as np
from mlsys.scorers import rel_err

def _reference(A):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    total = np.sum(S**2)
    if total == 0:
        return np.zeros_like(S, dtype=np.float64)
    var_exp = (S**2) / total
    return np.cumsum(var_exp)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    max_err = 0.0
    for n,d in [(5,3),(10,7),(8,12),(4,4)]:
        A = rng.standard_normal((n,d))
        try:
            cand = sol.read_singular_values_variance_explained(A)
        except Exception:
            return {"rel_err": float("inf")}
        ref = _reference(A)
        err = rel_err(ref, cand)
        if err > max_err:
            max_err = err
    return {"rel_err": max_err}
