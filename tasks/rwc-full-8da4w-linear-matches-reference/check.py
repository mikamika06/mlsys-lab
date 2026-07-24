import numpy as np


def _oracle(X, W, group_size):
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)

    x_scales = np.max(np.abs(X), axis=1) / 127.0
    x_scales = np.where(x_scales == 0, 1.0, x_scales)
    Xq = np.round(X / x_scales[:, None]).clip(-128, 127).astype(np.int32)

    n, k = W.shape
    groups = k // group_size
    w_scales = np.empty((n, groups), dtype=np.float64)
    Wq = np.empty((n, k), dtype=np.int32)

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

    accum = Xq @ Wq.T
    out = np.empty((X.shape[0], n), dtype=np.float64)
    for i in range(X.shape[0]):
        for r in range(n):
            value = 0.0
            for j in range(k):
                value += Xq[i, j] * Wq[r, j] * w_scales[r, j // group_size]
            out[i, r] = x_scales[i] * value
    return out


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[0.5, -1.2, 2.7, 3.1], [1.0, 0.0, -2.0, 1.5]]),
            np.array([[1.2, -3.0, 0.5, 2.2], [-1.0, 4.0, 2.0, -0.7]]),
            2,
        ),
        (
            np.array([
                [1.0, -4.0, 2.0, 0.25, 3.0, -1.0],
                [-2.0, 1.5, 0.5, 4.0, -3.0, 2.0],
            ]),
            np.array([
                [2.0, -1.0, 3.0, 0.5, -2.0, 4.0],
                [1.0, 2.0, -4.0, 3.0, 1.5, -1.5],
                [-3.0, 0.5, 2.5, -2.0, 4.0, 1.0],
            ]),
            3,
        ),
    ]

    err = 0.0
    for X, W, g in cases:
        try:
            got = np.asarray(sol.linear_8da4w(X, W, g), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle(X, W, g)
        err = max(err, float(np.max(np.abs(got - ref))))
    return {"max_abs_err": err}
