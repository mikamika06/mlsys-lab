import numpy as np


def restore_awq_equivalence(X, W, s):
    # TODO: applies AWQ scaling to weights but forgets the inverse scaling
    # on activations, so the linear layer output changes.
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    W_scaled = W * s[None, :]
    return X @ W_scaled.T
