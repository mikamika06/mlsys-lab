import math
import numpy as np


def layernorm_vjp(x: np.ndarray, grad_y: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grad_y = np.asarray(grad_y, dtype=np.float64)

    N, D = x.shape
    out = np.empty((N, D), dtype=np.float64)

    for i in range(N):
        mean_s = 0.0
        for j in range(D):
            mean_s += float(x[i, j])
        mean = mean_s / D

        var_s = 0.0
        for j in range(D):
            diff = float(x[i, j]) - mean
            var_s += diff * diff
        var = var_s / D

        inv_std = 1.0 / math.sqrt(var + eps)

        mean_grad_s = 0.0
        for j in range(D):
            mean_grad_s += float(grad_y[i, j])
        mean_grad = mean_grad_s / D

        mean_grad_hat_s = 0.0
        for j in range(D):
            x_hat_j = (float(x[i, j]) - mean) * inv_std
            mean_grad_hat_s += float(grad_y[i, j]) * x_hat_j
        mean_grad_hat = mean_grad_hat_s / D

        for j in range(D):
            x_hat_j = (float(x[i, j]) - mean) * inv_std
            out[i, j] = inv_std * (float(grad_y[i, j]) - mean_grad - x_hat_j * mean_grad_hat)

    return out
