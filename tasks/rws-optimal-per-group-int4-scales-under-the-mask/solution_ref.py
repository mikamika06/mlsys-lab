import numpy as np


def _naive_group_dequant(seg, qmax):
    amax = 0.0
    for val in seg:
        abs_val = val if val >= 0 else -val
        if abs_val > amax:
            amax = abs_val
    amax = float(amax)
    scale = amax / qmax if amax > 0 else 1.0
    dq_list = []
    for val in seg:
        r = round(val / scale)
        if r < -qmax:
            r = -qmax
        elif r > qmax:
            r = qmax
        dq_list.append(scale * r)
    return scale, np.array(dq_list, dtype=np.float64)


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
            amax = 0.0
            for val in seg:
                abs_val = val if val >= 0 else -val
                if abs_val > amax:
                    amax = abs_val
            amax = float(amax)
            best_err = float('inf')
            best_scale = None
            best_dq = None
            for alpha in alphas:
                scale = (alpha * amax / qmax) if amax > 0 else 1.0
                dq_list = []
                for val in seg:
                    r = round(val / scale)
                    if r < -qmax:
                        r = -qmax
                    elif r > qmax:
                        r = qmax
                    dq_list.append(scale * r)
                dq = np.array(dq_list, dtype=np.float64)
                trial = row_what.copy()
                trial[sl] = dq
                
                diff = row_target - trial
                N = X.shape[0]
                err_sum = 0.0
                for n in range(N):
                    dot_val = 0.0
                    for i_idx in range(I):
                        dot_val += X[n, i_idx] * diff[i_idx]
                    err_sum += dot_val * dot_val
                err = float(err_sum)
                
                if err < best_err:
                    best_err = err
                    best_scale = scale
                    best_dq = dq
            row_what[sl] = best_dq
            group_scales[o, g] = best_scale
        what[o] = row_what

    N = X.shape[0]
    total_squared_diff = 0.0
    total_elements = N * O
    for n in range(N):
        for o in range(O):
            y_val = 0.0
            for i_idx in range(I):
                y_val += X[n, i_idx] * Wm[o, i_idx]
            yhat_val = 0.0
            for i_idx in range(I):
                yhat_val += X[n, i_idx] * what[o, i_idx]
            diff_val = y_val - yhat_val
            total_squared_diff += diff_val * diff_val
    mse = float(total_squared_diff / total_elements)

    return group_scales, mse
