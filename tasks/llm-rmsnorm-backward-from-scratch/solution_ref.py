import numpy as np


def rmsnorm_backward(x: np.ndarray, grad_y: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    grad_y = np.asarray(grad_y, dtype=np.float64)
    d = x.size
    r = np.sqrt(np.mean(x * x) + eps)
    dot = np.sum(grad_y * x)
    dx = grad_y / r - x * dot / (d * r * r * r)
    return dx.astype(np.float64)
