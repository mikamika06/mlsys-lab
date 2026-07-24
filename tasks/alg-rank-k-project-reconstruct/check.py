import numpy as np
from mlsys.scorers import mse

def _reference(X: np.ndarray, k: int) -> np.ndarray:
    """Compute the rank‑k PCA reconstruction using NumPy SVD."""
    Xc = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vk = Vt[:k, :].T  # shape (d, k)
    Y = Xc @ Vk       # projection onto subspace
    X_hat = Y @ Vk.T + np.mean(X, axis=0)  # add mean back
    return X_hat.astype(np.float64)

def grade(sol, fx) -> dict:
    cases = [
        (np.random.randn(10, 5), 2),
        (np.random.randn(20, 3), 1),
        (np.random.randn(15, 8), 4),
        (np.random.randn(50, 10), 5),
        (np.random.randn(30, 7), 3)
    ]
    max_err = 0.0
    for X, k in cases:
        try:
            got = sol.rank_k_project_reconstruct(X, k)
        except Exception:
            return {"mse": float("inf")}
        ref = _reference(X, k)
        err = mse(ref, got)
        if err > max_err:
            max_err = err
    return {"mse": max_err}
