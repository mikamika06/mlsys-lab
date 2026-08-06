import math
import numpy as np


def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-free logistic sigmoid, evaluated branchlessly."""
    arr = np.asarray(x, dtype=np.float64)
    out = np.empty(arr.shape, dtype=np.float64)
    arr_flat = arr.flat
    out_flat = out.flat
    for i in range(arr.size):
        val = float(arr_flat[i])
        abs_val = val if val >= 0.0 else -val
        z = math.exp(-abs_val)
        denom = 1.0 + z
        if val >= 0.0:
            out_flat[i] = 1.0 / denom
        else:
            out_flat[i] = z / denom
    return out
