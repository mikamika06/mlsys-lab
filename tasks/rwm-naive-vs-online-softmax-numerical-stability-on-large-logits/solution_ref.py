import math
import numpy as np


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    rows = x.shape[0]
    cols = x.shape[1]
    out = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        max_val = x[i, 0]
        for j in range(1, cols):
            if x[i, j] > max_val:
                max_val = x[i, j]
        row_exp_sum = 0.0
        for j in range(cols):
            val = math.exp(x[i, j] - max_val)
            out[i, j] = val
            row_exp_sum += val
        for j in range(cols):
            out[i, j] /= row_exp_sum
    return out
