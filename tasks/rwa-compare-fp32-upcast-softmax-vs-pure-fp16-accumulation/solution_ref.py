import math
import numpy as np


def softmax_fp32(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    rows, cols = y.shape
    out = np.empty((rows, cols), dtype=np.float32)

    for i in range(rows):
        max_val = y[i, 0]
        for j in range(1, cols):
            if y[i, j] > max_val:
                max_val = y[i, j]

        row_sum = 0.0
        for j in range(cols):
            val = math.exp(float(y[i, j] - max_val))
            out[i, j] = val
            row_sum += val

        for j in range(cols):
            out[i, j] = out[i, j] / row_sum

    return out
