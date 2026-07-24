import numpy as np
from mlsys.scorers import rel_err

def _reference_inverse(A, lam):
    n = A.shape[0]
    H = A @ A.T + lam * np.eye(n, dtype=np.float64)
    L = np.linalg.cholesky(H)
    I = np.eye(n, dtype=np.float64)
    X = np.linalg.solve(L, I)
    inv_H = np.linalg.solve(L.T, X)
    return inv_H

def grade(sol, fx) -> dict:
    func = getattr(sol, "reconstruct_inverse_hessian", None)
    if not callable(func):
        return {"rel_err": float("inf")}
    rng = np.random.default_rng(42)
    for _ in range(5):
        n = rng.integers(3, 10)
        d = rng.integers(2, 8)
        A = rng.standard_normal((n, d))
        lam = rng.uniform(0.1, 1.0)
        try:
            cand = func(A, float(lam))
        except Exception:
            return {"rel_err": float("inf")}
        ref = _reference_inverse(A, lam)
        if cand.shape != ref.shape or cand.dtype != np.float64:
            return {"rel_err": float("inf")}
        err = rel_err(ref, cand)
        if err > 1e-4:
            return {"rel_err": err}
    # If all tests passed, return the last computed error (should be <= threshold)
    return {"rel_err": err}
