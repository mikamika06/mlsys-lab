import math
import numpy as np


def apply_rope(x: np.ndarray, position: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[0]
    half = d // 2

    out = np.empty_like(x, dtype=np.float64)
    for i in range(half):
        theta = position * (10000.0 ** (-2.0 * float(i) / d))
        c = math.cos(theta)
        s = math.sin(theta)
        a = x[i]
        b = x[i + half]
        out[i] = a * c - b * s
        out[i + half] = a * s + b * c
    return out
