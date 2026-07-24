import numpy as np
from mlsys.scorers import channel_rel_err

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        (rng.standard_normal((5, 3)), 2),
        (rng.standard_normal((10, 4)), 3),
        (rng.standard_normal((20, 6)), 4),
        (rng.standard_normal((50, 10)), 5),
        (rng.standard_normal((100, 15)), 7),
    ]
    max_err = 0.0
    for X, k in cases:
        try:
            got = sol.pca_power_iteration(X, k)
        except Exception:
            return {"channel_rel_err": float("inf")}
        if not isinstance(got, np.ndarray):
            return {"channel_rel_err": float("inf")}
        if got.shape != (k, X.shape[1]):
            return {"channel_rel_err": float("inf")}
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        ref = Vt[:k, :]
        for i in range(k):
            if np.dot(ref[i], got[i]) < 0:
                got[i] *= -1
        err = channel_rel_err(ref, got)
        max_err = max(max_err, err)
    return {"channel_rel_err": max_err}
