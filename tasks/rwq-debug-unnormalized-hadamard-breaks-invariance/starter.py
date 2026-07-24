import numpy as np


def _hadamard(n):
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H


def hadamard_rotate(X, W):
    # TODO: missing the 1/sqrt(n) normalization.
    # The raw Hadamard matrix satisfies H H^T = nI, so it is not a
    # length-preserving rotation.
    n = X.shape[1]
    H = _hadamard(n)
    return (X.astype(np.float64) @ H, H.T @ W.astype(np.float64))
