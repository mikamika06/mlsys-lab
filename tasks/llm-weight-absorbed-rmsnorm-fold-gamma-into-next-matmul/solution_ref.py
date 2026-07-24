import numpy as np


def fold_rmsnorm_gamma(W, b, gamma):
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    return gamma.reshape(1, -1) * W, b.copy()
