import math
import numpy as np


def log_condition_number(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x, dtype=np.float64)
    for index in np.ndindex(x.shape):
        val = x[index]
        res = math.log(val)
        res = 1.0 / res
        if res < 0.0:
            res = -res
        out[index] = res
    return out
