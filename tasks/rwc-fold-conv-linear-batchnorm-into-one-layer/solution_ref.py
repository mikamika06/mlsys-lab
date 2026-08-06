import math
import numpy as np


def fold_bn_into_linear(W, b, gamma, beta, running_mean, running_var, eps):
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    running_mean = np.asarray(running_mean, dtype=np.float64)
    running_var = np.asarray(running_var, dtype=np.float64)

    out_f = W.shape[0]
    in_f = W.shape[1]

    scale = np.zeros(out_f, dtype=np.float64)
    for i in range(out_f):
        scale[i] = gamma[i] / math.sqrt(running_var[i] + eps)

    W_folded = np.zeros((out_f, in_f), dtype=np.float64)
    for i in range(out_f):
        for j in range(in_f):
            W_folded[i, j] = W[i, j] * scale[i]

    b_folded = np.zeros(out_f, dtype=np.float64)
    for i in range(out_f):
        b_folded[i] = scale[i] * (b[i] - running_mean[i]) + beta[i]

    return W_folded, b_folded
