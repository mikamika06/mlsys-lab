import numpy as np

def pairwise_mahalanobis(X: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    Y = X @ cov_inv
    XY = X @ Y.T
    diag = np.einsum('ij,ij->i', X, Y)
    D2 = diag[:, None] + diag[None, :] - 2 * XY
    return np.sqrt(np.maximum(D2, 0))
