import numpy as np


def softmax_kernel(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    m = np.max(X, axis=1, keepdims=True)
    e = np.exp(X - m)
    return e / np.sum(e, axis=1, keepdims=True)
