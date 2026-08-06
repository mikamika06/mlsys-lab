import math
import numpy as np


def stable_one_minus_cos_over_x2(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty(x_arr.shape, dtype=np.float64)
    for idx in np.ndindex(x_arr.shape):
        val = float(x_arr[idx])
        half = val / 2.0
        out[idx] = 0.5 * (math.sin(half) / half) ** 2
    return out
