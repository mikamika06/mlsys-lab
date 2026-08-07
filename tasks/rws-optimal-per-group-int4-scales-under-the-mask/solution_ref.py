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
    return scale, dq_list


def optimal_group_scales_under_mask(W: list[list[float]], M: list[list[float]], X: list[list[float]],
                                     group_size: int, bits: int = 4,
                                     alphas: list[float] = None):
    """Greedy per-row, per-group coordinate-descent scale search
    minimizing the X-weighted output MSE of a masked-then-quantized
    weight matrix. See task.md for the exact algorithm.
    """
    if alphas is None:
        alphas = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    else:
        alphas = [float(a) for a in alphas]

    qmax = (1 << (bits - 1)) - 1
    O = len(W)
    I = len(W[0])
    n_groups = I // group_size

    Wm = [[float(W[o][i]) * float(M[o][i]) for i in range(I)] for o in range(O)]

    what = [[0.0] * I for _ in range(O)]
    for o in range(O):
        for g in range(n_groups):
            start = g * group_size
            end = (g + 1) * group_size
            seg = Wm[o][start:end]
            _s, dq = _naive_group_dequant(seg, qmax)
            for idx_in_seg, val in enumerate(dq):
                what[o][start + idx_in_seg] = val

    group_scales = [[0.0] * n_groups for _ in range(O)]
    for o in range(O):
        row_target = Wm[o]
        row_what = list(what[o])
        for g in range(n_groups):
            start = g * group_size
            end = (g + 1) * group_size
            seg = row_target[start:end]
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

                trial = list(row_what)
                for idx_in_seg, val in enumerate(dq_list):
                    trial[start + idx_in_seg] = val

                N = len(X)
                err_sum = 0.0
                for n in range(N):
                    dot_val = 0.0
                    for i_idx in range(I):
                        diff_val = row_target[i_idx] - trial[i_idx]
                        dot_val += X[n][i_idx] * diff_val
                    err_sum += dot_val * dot_val
                err = float(err_sum)

                if err < best_err:
                    best_err = err
                    best_scale = scale
                    best_dq = dq_list

            for idx_in_seg, val in enumerate(best_dq):
                row_what[start + idx_in_seg] = val
            group_scales[o][g] = best_scale
        what[o] = row_what

    N = len(X)
    total_squared_diff = 0.0
    total_elements = N * O
    for n in range(N):
        for o in range(O):
            y_val = 0.0
            for i_idx in range(I):
                y_val += X[n][i_idx] * Wm[o][i_idx]
            yhat_val = 0.0
            for i_idx in range(I):
                yhat_val += X[n][i_idx] * what[o][i_idx]
            diff_val = y_val - yhat_val
            total_squared_diff += diff_val * diff_val
    mse = float(total_squared_diff / total_elements)

    return group_scales, mse
