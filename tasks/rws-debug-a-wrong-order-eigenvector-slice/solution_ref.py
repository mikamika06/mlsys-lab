import numpy as np


def pca_reconstruct(X: np.ndarray, k: int) -> np.ndarray:
    """
    Rank-k reconstruction of X via the top-k principal directions of
    C = X^T X / n (uncentered covariance / Gram matrix).

    np.linalg.eigh returns eigenvalues in ASCENDING order, so the
    largest-eigenvalue (most-variance) directions are the LAST k columns
    of the returned eigenvector matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    C = X.T @ X / n
    eigvals, eigvecs = np.linalg.eigh(C)
    V = eigvecs[:, -k:]  # largest-eigenvalue directions
    P = X @ V
    return P @ V.T
