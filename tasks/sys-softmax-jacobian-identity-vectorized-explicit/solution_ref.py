import numpy as np


def softmax_jacobian_vjp(p: np.ndarray, dY: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    dY = np.asarray(dY, dtype=np.float64)
    dot = np.sum(p * dY, axis=1, keepdims=True)
    return p * (dY - dot)
