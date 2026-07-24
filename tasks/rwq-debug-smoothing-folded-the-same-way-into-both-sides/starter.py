import numpy as np


def fold_smoothing(X, W, s):
    # TODO: this incorrectly applies smoothing to both factors.
    # The second factor must receive the inverse scale.
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    return X * s.reshape(1, -1), W * s.reshape(-1, 1)
