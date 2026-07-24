import numpy as np


def cholesky_lower(A: np.ndarray) -> np.ndarray:
    """Cholesky-Banachiewicz: A = L L^T with L lower triangular, L_ii > 0."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        s = A[j, j] - L[j, :j] @ L[j, :j]
        if s <= 0.0:
            raise np.linalg.LinAlgError("matrix is not positive definite")
        L[j, j] = np.sqrt(s)
        if j + 1 < n:
            L[j + 1:, j] = (A[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]

    return L
