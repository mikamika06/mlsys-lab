import numpy as np


def imatrix_best_scale(x: np.ndarray, w: np.ndarray, scale_grid: np.ndarray, qmin: int, qmax: int) -> int:
    """
    Sweep the candidate scale grid; for each scale, symmetric-quantize x
    (round(x/s), clipped to [qmin, qmax]), dequantize, and score with the
    imatrix-weighted squared error sum(w * (x - xhat)**2). Return the index
    of the grid entry with the smallest weighted error (first on ties).
    """
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    grid = np.asarray(scale_grid, dtype=np.float64)

    best_i = 0
    best_err = np.inf
    for i, s in enumerate(grid):
        q = np.clip(np.round(x / s), qmin, qmax)
        xhat = q * s
        err = float(np.sum(w * (x - xhat) ** 2))
        if err < best_err:
            best_err = err
            best_i = i
    return best_i
