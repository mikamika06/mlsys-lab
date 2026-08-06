import math
import numpy as np


def softmax(x):
    """Numerically stable softmax computation."""
    arr = np.asarray(x, dtype=np.float64)
    n = arr.size
    if n == 0:
        return np.empty_like(arr)

    max_val = float(arr.flat[0])
    for i in range(1, n):
        val = float(arr.flat[i])
        if val > max_val:
            max_val = val

    exp_vals = [0.0] * n
    sum_e = 0.0
    for i in range(n):
        ev = math.exp(float(arr.flat[i]) - max_val)
        exp_vals[i] = ev
        sum_e += ev

    out = np.empty(arr.shape, dtype=np.float64)
    for i in range(n):
        out.flat[i] = exp_vals[i] / sum_e

    return out
