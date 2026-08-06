import math
import numpy as np


def rmsnorm_backward(x: np.ndarray, grad_y: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grad_y = np.asarray(grad_y, dtype=np.float64)
    d = x.size

    sum_sq = 0.0
    for i in range(d):
        val = x[i]
        sum_sq += val * val

    r = math.sqrt((sum_sq / d) + eps)

    dot = 0.0
    for i in range(d):
        dot += grad_y[i] * x[i]

    dx = np.empty(d, dtype=np.float64)
    denom = d * r * r * r
    for i in range(d):
        dx[i] = grad_y[i] / r - x[i] * dot / denom

    return dx
