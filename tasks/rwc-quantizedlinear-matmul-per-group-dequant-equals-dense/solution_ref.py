import numpy as np


def quantized_linear(X, Q, scales, zeros, group_size):
    m = Q.shape[0]
    groups = np.arange(m) // group_size
    W = (Q.astype(np.float64) - np.asarray(zeros, dtype=np.float64)[groups, None])
    W = W * np.asarray(scales, dtype=np.float64)[groups, None]
    return np.asarray(X, dtype=np.float64) @ W.T
