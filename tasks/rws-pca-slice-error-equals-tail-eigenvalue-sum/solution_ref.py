import numpy as np


def pca_slice_error(X: np.ndarray, k: int) -> float:
    """Squared Frobenius error of projecting X onto the top-k eigenvectors
    of the Gram matrix G = X^T X (no centering): ||X - X Q_k Q_k^T||_F^2.
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    G = X.T @ X
    eigvals_asc, eigvecs_asc = np.linalg.eigh(G)  # ascending eigenvalue order

    if k <= 0:
        Xrec = np.zeros_like(X)
    else:
        # top-k eigenvectors are the last k columns (largest eigenvalues)
        Qk = eigvecs_asc[:, d - k:]
        Xrec = X @ Qk @ Qk.T

    return float(np.sum((X - Xrec) ** 2))
