import numpy as np


def migrate_scale(X: np.ndarray, W: np.ndarray, s: np.ndarray):
    X_new = np.asarray(X, dtype=np.float64) * np.asarray(s, dtype=np.float64)[None, :]
    W_new = np.asarray(W, dtype=np.float64) / np.asarray(s, dtype=np.float64)[:, None]
    return X_new, W_new
