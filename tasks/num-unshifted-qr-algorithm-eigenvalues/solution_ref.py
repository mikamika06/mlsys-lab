import numpy as np


def qr_eigenvalues(A: np.ndarray, max_iter: int = 1000, tol: float = 1e-12) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64).copy()
    for _ in range(max_iter):
        Q, R = np.linalg.qr(A)
        A = R @ Q
        off = A - np.diag(np.diag(A))
        if np.linalg.norm(off) < tol:
            break
    return np.diag(A).astype(np.float64)
