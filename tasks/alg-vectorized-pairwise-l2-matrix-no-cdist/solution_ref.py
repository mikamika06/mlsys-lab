import numpy as np

def pairwise_l2_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X_norm = (X**2).sum(axis=1)[:, None]
    Y_norm = (Y**2).sum(axis=1)[None, :]
    return X_norm + Y_norm - 2 * X.dot(Y.T)
