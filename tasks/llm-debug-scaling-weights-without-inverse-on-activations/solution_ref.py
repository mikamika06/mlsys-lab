import numpy as np


def restore_awq_equivalence(X, W, s):
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    W_scaled = W * s[None, :]
    X_compensated = X / s[None, :]
    return X_compensated @ W_scaled.T
