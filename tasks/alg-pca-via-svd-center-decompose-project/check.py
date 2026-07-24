import numpy as np
from mlsys.scorers import channel_rel_err

def _reference(X, k):
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    centered = X - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:k].T

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        (np.array([[1., 2.], [3., 4.], [5., 6.]]), 1),
        (rng.random((10, 5)), 3),
        (rng.random((20, 8)), 5),
        (np.arange(12).reshape(3, 4), 2)
    ]
    for X, k in cases:
        try:
            out = sol.pca_svd(X, k)
        except Exception:
            return {"channel_rel_err": float("inf")}
        ref = _reference(X, k)
        if out.shape != ref.shape:
            return {"channel_rel_err": float("inf")}
        aligned = np.array(out, copy=True)
        for j in range(ref.shape[1]):
            dot = np.sum(aligned[:, j] * ref[:, j])
            if dot < 0:
                aligned[:, j] *= -1
        err = channel_rel_err(ref, aligned)
        if err > 1e-6:
            return {"channel_rel_err": err}
    return {"channel_rel_err": 0.0}
