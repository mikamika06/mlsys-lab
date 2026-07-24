import numpy as np


def cholesky_inverse(H: np.ndarray) -> np.ndarray:
    """
    Invert an SPD matrix H via its Cholesky factor: H = L L^T, then
    H^-1 = (L^-1)^T (L^-1). Never forms H^-1 through a generic elimination
    path; only ever solves against the triangular factor L.
    """
    H = np.asarray(H, dtype=np.float64)
    n = H.shape[0]
    L = np.linalg.cholesky(H)
    L_inv = np.linalg.solve(L, np.eye(n))
    return L_inv.T @ L_inv
