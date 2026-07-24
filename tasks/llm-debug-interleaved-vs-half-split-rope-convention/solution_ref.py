import numpy as np


def apply_rope(x: np.ndarray, position: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[0]
    half = d // 2

    idx = np.arange(half, dtype=np.float64)
    theta = position * (10000.0 ** (-2.0 * idx / d))
    c = np.cos(theta)
    s = np.sin(theta)

    out = np.empty_like(x, dtype=np.float64)
    a = x[:half]
    b = x[half:]
    out[:half] = a * c - b * s
    out[half:] = a * s + b * c
    return out
