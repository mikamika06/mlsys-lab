import numpy as np

def cov_and_eig(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the empirical covariance matrix of X and its eigen decomposition.
    The eigenvalues are sorted in descending order; eigenvectors correspondingly.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    C = (X.T @ X) / n
    w, v = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]          # descending order
    return C, w[idx], v[:, idx]
