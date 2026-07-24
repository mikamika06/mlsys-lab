import numpy as np


def percentile_clip_best(x: np.ndarray, percentile_grid: np.ndarray, qmax: int):
    """
    Sweep candidate clip percentiles; for each, clip x to the percentile of
    |x|, symmetric-quantize at that range with qmax codes, dequantize, and
    score by MSE against the (unclipped) original x. Return the
    (index, mse) of the grid entry with the smallest MSE.
    """
    x = np.asarray(x, dtype=np.float64)
    grid = np.asarray(percentile_grid, dtype=np.float64)

    best_i = 0
    best_mse = np.inf
    for i, p in enumerate(grid):
        thr = float(np.percentile(np.abs(x), p))
        if thr <= 0.0:
            thr = 1e-8
        clipped = np.clip(x, -thr, thr)
        scale = thr / qmax
        codes = np.clip(np.round(clipped / scale), -qmax, qmax)
        deq = codes * scale
        mse = float(np.mean((x - deq) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_i = i
    return best_i, best_mse
