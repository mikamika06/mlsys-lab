import numpy as np


def _naive_group_dequant(seg, qmax):
    amax = float(np.max(np.abs(seg)))
    scale = amax / qmax if amax > 0 else 1.0
    dequant = scale * np.clip(np.round(seg / scale), -qmax, qmax)
    return scale, dequant


def optimal_group_scales_under_mask(W: np.ndarray, M: np.ndarray, X: np.ndarray,
                                     group_size: int, bits: int = 4,
                                     alphas: np.ndarray = None):
    """Greedy per-row, per-group coordinate-descent scale search
    minimizing the X-weighted output MSE of a masked-then-quantized
    weight matrix. See task.md for the exact algorithm.
    """
    if alphas is None:
        alphas = np.linspace(0.6, 1.4, 9)
    alphas = np.asarray(alphas, dtype=np.float64)

    W = np.asarray(W, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    qmax = (1 << (bits - 1)) - 1
    Wm = W * M
    O, I = Wm.shape
    n_groups = I // group_size

    what = np.zeros_like(Wm)
    for o in range(O):
        for g in range(n_groups):
            sl = slice(g * group_size, (g + 1) * group_size)
            _s, dq = _naive_group_dequant(Wm[o, sl], qmax)
            what[o, sl] = dq

    group_scales = np.zeros((O, n_groups), dtype=np.float64)
    for o in range(O):
        row_target = Wm[o]
        row_what = what[o].copy()
        for g in range(n_groups):
            sl = slice(g * group_size, (g + 1) * group_size)
            seg = row_target[sl]
            amax = float(np.max(np.abs(seg)))
            best_err = np.inf
            best_scale = None
            best_dq = None
            for alpha in alphas:
                scale = (alpha * amax / qmax) if amax > 0 else 1.0
                dq = scale * np.clip(np.round(seg / scale), -qmax, qmax)
                trial = row_what.copy()
                trial[sl] = dq
                err = float(np.sum((X @ (row_target - trial)) ** 2))
                if err < best_err:
                    best_err = err
                    best_scale = scale
                    best_dq = dq
            row_what[sl] = best_dq
            group_scales[o, g] = best_scale
        what[o] = row_what

    Y = X @ Wm.T
    Yhat = X @ what.T
    mse = float(np.mean((Y - Yhat) ** 2))

    return group_scales, mse
