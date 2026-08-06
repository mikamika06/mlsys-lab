import math
import numpy as np


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    b, d = x.shape
    out = np.empty((b, d), dtype=np.float64)
    eps = 1e-5

    for i in range(b):
        total = 0.0
        for j in range(d):
            total += x[i, j]
        mean = total / d

        var_sum = 0.0
        for j in range(d):
            diff = x[i, j] - mean
            var_sum += diff * diff
        var = var_sum / d

        std = math.sqrt(var + eps)

        for j in range(d):
            x_hat = (x[i, j] - mean) / std
            g = gamma[j] if gamma.ndim == 1 else gamma[0, j]
            b_val = beta[j] if beta.ndim == 1 else beta[0, j]
            out[i, j] = g * x_hat + b_val

    return out
