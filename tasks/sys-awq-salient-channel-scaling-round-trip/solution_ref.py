import math
import numpy as np


def awq_matmul(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    rows, cols = W.shape
    x_rows, x_cols = X.shape

    scale = np.empty((1, cols), dtype=np.float64)
    for j in range(cols):
        max_val = 0.0
        for i in range(rows):
            val = W[i, j]
            abs_val = val if val >= 0.0 else -val
            if abs_val > max_val:
                max_val = abs_val
        s = max_val / 7.0
        if s == 0.0:
            scale[0, j] = 1.0
        else:
            scale[0, j] = s

    q = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            val = W[i, j] / scale[0, j]
            rounded = math.floor(val + 0.5) if val >= 0.0 else math.ceil(val - 0.5)
            if rounded < -8.0:
                q[i, j] = -8.0
            elif rounded > 7.0:
                q[i, j] = 7.0
            else:
                q[i, j] = rounded

    W_dequant = np.empty((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            W_dequant[i, j] = q[i, j] * scale[0, j]

    out_rows = rows
    out_cols = x_cols
    result = np.empty((out_rows, out_cols), dtype=np.float64)

    for i in range(out_rows):
        for j in range(out_cols):
            acc = 0.0
            for k in range(cols):
                acc += W_dequant[i, k] * X[k, j]
            result[i, j] = acc

    return result
