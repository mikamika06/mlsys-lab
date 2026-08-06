import math
import numpy as np


def per_token_int8_dequant(X):
    X = np.asarray(X, dtype=np.float64)
    rows, cols = X.shape
    out = np.zeros((rows, cols), dtype=np.float64)

    for i in range(rows):
        max_abs = 0.0
        for j in range(cols):
            val = abs(X[i, j])
            if val > max_abs:
                max_abs = val
        scale = max_abs / 127.0

        for j in range(cols):
            if scale != 0:
                val = X[i, j] / scale
                q = int(math.floor(val + 0.5) if val >= 0 else math.ceil(val - 0.5))
            else:
                q = 0
            out[i, j] = float(np.int8(q)) * scale

    return out
