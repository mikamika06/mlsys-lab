import numpy as np


def softmax_vjp(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    shifted = x - np.max(x)
    e = np.exp(shifted)
    s = e / np.sum(e)
    return s * (g - np.sum(g * s))
