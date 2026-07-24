import numpy as np

def zca_whitening(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    X_centered = X - mean
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    inv_sqrt_eigvals = np.diag(1.0 / np.sqrt(eigvals))
    W_zca = eigvecs @ inv_sqrt_eigvals @ eigvecs.T
    X_whitened = X_centered @ W_zca
    return X_whitened
