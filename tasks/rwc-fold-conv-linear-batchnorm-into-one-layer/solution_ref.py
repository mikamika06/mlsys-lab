import numpy as np


def fold_bn_into_linear(W, b, gamma, beta, running_mean, running_var, eps):
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    running_mean = np.asarray(running_mean, dtype=np.float64)
    running_var = np.asarray(running_var, dtype=np.float64)

    scale = gamma / np.sqrt(running_var + eps)
    W_folded = W * scale[:, None]
    b_folded = scale * (b - running_mean) + beta
    return W_folded, b_folded
