import numpy as np

def mds_from_distances(D2: np.ndarray, k: int) -> np.ndarray:
    """
    Classical Multidimensional Scaling.

    Parameters
    ----------
    D2 : np.ndarray
        Squared Euclidean distance matrix of shape (n, n).
    k : int
        Target dimensionality of the embedding.

    Returns
    -------
    X : np.ndarray
        Coordinates of shape (n, k) in float64.
    """
    if D2.ndim != 2 or D2.shape[0] != D2.shape[1]:
        raise ValueError("D2 must be a square matrix")
    n = D2.shape[0]
    J = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    lam = eigvals[idx][:k]
    vec = eigvecs[:, idx[:k]]
    return vec * np.sqrt(np.maximum(lam, 0))
