import numpy as np


def obq_single_weight_update(w, Hinv, grid):
    w = np.asarray(w, dtype=np.float64)
    Hinv = np.asarray(Hinv, dtype=np.float64)
    grid = np.asarray(grid, dtype=np.float64)

    nearest = grid[np.argmin(np.abs(w[:, None] - grid[None, :]), axis=1)]
    costs = ((nearest - w) ** 2) / np.diag(Hinv)
    k = int(np.argmin(costs))

    q = nearest[k]
    err = q - w[k]

    out = w.copy()
    out -= err * Hinv[:, k] / Hinv[k, k]
    out[k] = q
    return out
