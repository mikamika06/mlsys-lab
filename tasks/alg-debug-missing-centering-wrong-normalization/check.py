import numpy as np
from mlsys.scorers import max_abs_err

def _reference_eigenvector(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    ref = Vt[0]
    return ref / np.linalg.norm(ref)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    errors = []
    for n, d in [(50, 5), (100, 10), (200, 20)]:
        X = rng.standard_normal((n, d))
        try:
            v = sol.leading_eigenvector(X, num_iter=500)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _reference_eigenvector(X)
        if np.dot(v, ref) < 0:
            v = -v
        errors.append(max_abs_err(ref, v))
    max_err = max(errors)
    return {"max_abs_err": max_err}
