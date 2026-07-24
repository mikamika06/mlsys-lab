import numpy as np


def _hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def hadamard_rotate(X, W):
    n = X.shape[1]
    H = _hadamard(n)
    Q = H / np.sqrt(n)
    return (X.astype(np.float64) @ Q, Q.T @ W.astype(np.float64))
