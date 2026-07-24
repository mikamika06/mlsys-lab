import numpy as np


def linear_8da4w(X: np.ndarray, W: np.ndarray, group_size: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    x_scales = np.max(np.abs(X), axis=1) / 127.0
    x_scales = np.where(x_scales == 0, 1.0, x_scales)
    Xq = np.round(X / x_scales[:, None]).clip(-128, 127).astype(np.int32)

    n, k = W.shape
    groups = k // group_size
    Wq = np.empty((n, k), dtype=np.int32)
    w_scales = np.empty((n, groups), dtype=np.float64)

    for r in range(n):
        for t in range(groups):
            start = t * group_size
            end = start + group_size
            scale = np.max(np.abs(W[r, start:end])) / 7.0
            if scale == 0:
                scale = 1.0
            w_scales[r, t] = scale
            Wq[r, start:end] = np.round(
                W[r, start:end] / scale
            ).clip(-8, 7).astype(np.int32)

    out = np.zeros((X.shape[0], n), dtype=np.float64)
    for i in range(X.shape[0]):
        for r in range(n):
            total = 0.0
            for j in range(k):
                total += Xq[i, j] * Wq[r, j] * w_scales[r, j // group_size]
            out[i, r] = x_scales[i] * total
    return out
