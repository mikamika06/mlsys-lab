import numpy as np
from mlsys.scorers import rel_err

def _ref_saliency(W, A):
    return np.sum(A**2, axis=0) * np.sum(W**2, axis=0)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_err = 0.0
    for shape in [(10, 5), (20, 8), (15, 12)]:
        n_out, n_in = shape
        W = rng.standard_normal((n_out, n_in)).astype(np.float64)
        A = rng.standard_normal((30, n_in)).astype(np.float64)  # batch size 30
        try:
            cand = sol.hessian_saliency(W.tolist(), A.tolist())
        except Exception:
            return {"rel_err": float("inf")}
        ref = _ref_saliency(W, A)
        err = rel_err(ref, cand)
        if err > max_err:
            max_err = err
    return {"rel_err": max_err}
