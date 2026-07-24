import numpy as np


def svd_singular_values(A: np.ndarray) -> np.ndarray:
    """Singular values of A via the eigendecomposition of the Gram matrix A^T A."""
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    k = min(m, n)
    G = A.T @ A
    w = np.linalg.eigvalsh(G)
    w = np.clip(w, 0.0, None)
    vals = np.sort(np.sqrt(w))[::-1]
    return vals[:k]
