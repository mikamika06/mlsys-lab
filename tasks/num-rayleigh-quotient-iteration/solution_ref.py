import numpy as np


def rayleigh_quotient_iteration(A: np.ndarray, v0: np.ndarray, n_iter: int) -> float:
    """Rayleigh quotient iteration: adaptive-shift inverse iteration, cubic convergence."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    v = np.asarray(v0, dtype=np.float64)
    v = v / np.linalg.norm(v)
    mu = float(v @ A @ v)

    for _ in range(n_iter):
        try:
            w = np.linalg.solve(A - mu * np.eye(n), v)
        except np.linalg.LinAlgError:
            w = v
        nrm = np.linalg.norm(w)
        if nrm == 0.0 or not np.isfinite(nrm):
            break
        v = w / nrm
        mu = float(v @ A @ v)

    return mu
