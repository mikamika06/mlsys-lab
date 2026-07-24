import numpy as np


def awq_matmul(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    scale = np.max(np.abs(W), axis=0, keepdims=True) / 7.0
    scale = np.where(scale == 0, 1.0, scale)

    q = np.round(W / scale)
    q = np.clip(q, -8, 7)

    W_dequant = q * scale
    return W_dequant @ X
