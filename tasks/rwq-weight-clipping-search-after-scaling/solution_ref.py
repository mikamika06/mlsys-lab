import numpy as np


def awq_clip_search(W: np.ndarray, group_size: int, clip_ratios: np.ndarray, bits: int = 4):
    W = np.asarray(W, dtype=np.float64)
    clip_ratios = np.asarray(clip_ratios, dtype=np.float64)

    rows, cols = W.shape
    ng = cols // group_size
    Wg = W.reshape(rows, ng, group_size)
    qmax = 2 ** (bits - 1) - 1
    n_ratios = clip_ratios.shape[0]

    amax = np.empty((rows, ng), dtype=np.float64)
    for i in range(rows):
        for g in range(ng):
            m = 0.0
            for k in range(group_size):
                val = abs(Wg[i, g, k])
                if val > m:
                    m = val
            amax[i, g] = m

    mse_grid = np.empty((rows, ng, n_ratios), dtype=np.float64)

    for ri in range(n_ratios):
        r = clip_ratios[ri]
        for i in range(rows):
            for g in range(ng):
                clipped_amax = amax[i, g] * r
                clipped_amax_safe = 1.0 if clipped_amax == 0.0 else clipped_amax
                scale = clipped_amax_safe / qmax

                group_sum = 0.0
                for k in range(group_size):
                    val = Wg[i, g, k]
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

                mse_grid[i, g, ri] = group_sum / group_size

    best_idx = np.empty((rows, ng), dtype=np.int64)
    best_mse = np.empty((rows, ng), dtype=np.float64)

    for i in range(rows):
        for g in range(ng):
            min_val = float('inf')
            best_ri = 0
            for ri in range(n_ratios):
                val = mse_grid[i, g, ri]
                if val < min_val:
                    min_val = val
                    best_ri = ri
            best_idx[i, g] = best_ri
            best_mse[i, g] = min_val

    return best_idx, best_mse
