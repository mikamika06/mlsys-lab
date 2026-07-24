import numpy as np


def awq_clip_search(W: np.ndarray, group_size: int, clip_ratios: np.ndarray, bits: int = 4):
    W = np.asarray(W, dtype=np.float64)
    clip_ratios = np.asarray(clip_ratios, dtype=np.float64)

    rows, cols = W.shape
    ng = cols // group_size
    Wg = W.reshape(rows, ng, group_size)
    amax = np.max(np.abs(Wg), axis=2)
    qmax = 2 ** (bits - 1) - 1

    n_ratios = clip_ratios.shape[0]
    mse_grid = np.empty((rows, ng, n_ratios), dtype=np.float64)

    for ri in range(n_ratios):
        r = clip_ratios[ri]
        clipped_amax = amax * r
        clipped_amax_safe = np.where(clipped_amax == 0, 1.0, clipped_amax)
        scale = clipped_amax_safe / qmax

        Wc = np.clip(Wg, -clipped_amax[:, :, None], clipped_amax[:, :, None])
        q = np.clip(np.round(Wc / scale[:, :, None]), -qmax, qmax)
        deq = q * scale[:, :, None]

        mse_grid[:, :, ri] = np.mean((Wg - deq) ** 2, axis=2)

    best_idx = np.argmin(mse_grid, axis=2).astype(np.int64)
    best_mse = np.min(mse_grid, axis=2)
    return best_idx, best_mse
