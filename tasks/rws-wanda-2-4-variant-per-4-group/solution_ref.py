import math
import numpy as np


def wanda_2_4_mask(W, X):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    m, n = W.shape
    x_rows = X.shape[0]

    scale = np.zeros(n, dtype=np.float64)
    for col in range(n):
        acc = 0.0
        for r in range(x_rows):
            val = X[r, col]
            acc += val * val
        mean_val = acc / x_rows
        scale[col] = math.sqrt(mean_val)

    scores = np.zeros((m, n), dtype=np.float64)
    for row in range(m):
        for col in range(n):
            w_val = W[row, col]
            if w_val < 0.0:
                abs_w = -w_val
            else:
                abs_w = w_val
            scores[row, col] = abs_w * scale[col]

    mask = np.zeros((m, n), dtype=np.int64)

    for row in range(m):
        for start in range(0, n, 4):
            vals = []
            for i in range(4):
                vals.append((scores[row, start + i], i))
            
            order = sorted(range(4), key=lambda i: (-vals[i][0], vals[i][1]))
            
            for idx in order[:2]:
                mask[row, start + idx] = 1

    return mask
