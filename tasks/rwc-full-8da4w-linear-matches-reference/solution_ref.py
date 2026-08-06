import numpy as np


def linear_8da4w(X: np.ndarray, W: np.ndarray, group_size: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    m, k = X.shape
    x_scales = np.empty(m, dtype=np.float64)
    Xq = np.empty((m, k), dtype=np.int32)

    for i in range(m):
        max_abs = 0.0
        for j in range(k):
            val = float(X[i, j])
            abs_val = val if val >= 0.0 else -val
            if abs_val > max_abs:
                max_abs = abs_val
        scale = max_abs / 127.0
        if scale == 0.0:
            scale = 1.0
        x_scales[i] = scale

        for j in range(k):
            v = round(float(X[i, j]) / scale)
            if v < -128:
                v = -128
            elif v > 127:
                v = 127
            Xq[i, j] = v

    n = W.shape[0]
    groups = k // group_size
    Wq = np.empty((n, k), dtype=np.int32)
    w_scales = np.empty((n, groups), dtype=np.float64)

    for r in range(n):
        for t in range(groups):
            start = t * group_size
            end = start + group_size
            max_abs = 0.0
            for j in range(start, end):
                val = float(W[r, j])
                abs_val = val if val >= 0.0 else -val
                if abs_val > max_abs:
                    max_abs = abs_val
            scale = max_abs / 7.0
            if scale == 0.0:
                scale = 1.0
            w_scales[r, t] = scale
            for j in range(start, end):
                v = round(float(W[r, j]) / scale)
                if v < -8:
                    v = -8
                elif v > 7:
                    v = 7
                Wq[r, j] = v

    out = np.zeros((m, n), dtype=np.float64)
    for i in range(m):
        for r in range(n):
            total = 0.0
            for j in range(k):
                total += float(Xq[i, j]) * float(Wq[r, j]) * float(w_scales[r, j // group_size])
            out[i, r] = x_scales[i] * total
    return out
