import numpy as np


def fix_awq_scale(W: np.ndarray, X: np.ndarray, s: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    W_scaled = W * s.reshape(1, -1)
    X_fixed = X / s.reshape(1, -1)
    return X_fixed @ W_scaled.T
