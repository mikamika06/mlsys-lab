def awq_clip_search(W: list[list[float]], group_size: int, clip_ratios: list[float], bits: int = 4) -> tuple[list[list[int]], list[list[float]]]:
    rows = len(W)
    cols = len(W[0])
    ng = cols // group_size
    qmax = 2 ** (bits - 1) - 1
    n_ratios = len(clip_ratios)

    amax = []
    for i in range(rows):
        row_amax = []
        for g in range(ng):
            m = 0.0
            for k in range(group_size):
                val = abs(W[i][g * group_size + k])
                if val > m:
                    m = val
            row_amax.append(m)
        amax.append(row_amax)

    mse_grid = []
    for i in range(rows):
        row_mse_grid = []
        for g in range(ng):
            row_mse_grid.append([0.0] * n_ratios)
        mse_grid.append(row_mse_grid)

    for ri in range(n_ratios):
        r = clip_ratios[ri]
        for i in range(rows):
            for g in range(ng):
                clipped_amax = amax[i][g] * r
                clipped_amax_safe = 1.0 if clipped_amax == 0.0 else clipped_amax
                scale = clipped_amax_safe / qmax

                group_sum = 0.0
                for k in range(group_size):
                    val = W[i][g * group_size + k]
                    if val < -clipped_amax:
                        Wc_val = -clipped_amax
                    elif val > clipped_amax:
                        Wc_val = clipped_amax
                    else:
                        Wc_val = val

                    ratio_val = Wc_val / scale
                    rounded_val = round(ratio_val)
                    if rounded_val < -qmax:
                        q_val = -qmax
                    elif rounded_val > qmax:
                        q_val = qmax
                    else:
                        q_val = rounded_val

                    deq_val = q_val * scale
                    group_sum += (val - deq_val) ** 2

                mse_grid[i][g][ri] = group_sum / group_size

    best_idx = []
    best_mse = []

    for i in range(rows):
        row_idx = []
        row_mse = []
        for g in range(ng):
            min_val = float('inf')
            best_ri = 0
            for ri in range(n_ratios):
                val = mse_grid[i][g][ri]
                if val < min_val:
                    min_val = val
                    best_ri = ri
            row_idx.append(best_ri)
            row_mse.append(min_val)
        best_idx.append(row_idx)
        best_mse.append(row_mse)

    return best_idx, best_mse
