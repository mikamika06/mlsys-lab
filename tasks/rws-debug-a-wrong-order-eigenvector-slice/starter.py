import numpy as np


def pca_reconstruct(X: np.ndarray, k: int) -> np.ndarray:
    """
    Rank-k reconstruction of X via the top-k principal directions of
    C = X^T X / n. Should return the reconstruction using the k
    directions of LARGEST variance (smallest reconstruction error).

    BUG: this keeps the k directions of *smallest* eigenvalue instead of
    largest -- fix the slice.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    C = X.T @ X / n
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending eigenvalue order
    order = np.argsort(eigvals)  # already ascending; re-sorting is a no-op
    V = eigvecs[:, order[:k]]  # BUG: keeps the SMALLEST-eigenvalue directions
    P = X @ V
    return P @ V.T
