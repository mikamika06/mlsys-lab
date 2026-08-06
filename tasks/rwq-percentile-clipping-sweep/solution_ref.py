import math
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

    n = x.shape[0]
    grid_len = grid.shape[0]

    abs_x = np.empty(n, dtype=np.float64)
    for i in range(n):
        val = x[i]
        abs_x[i] = -val if val < 0.0 else val

    sorted_abs = np.empty(n, dtype=np.float64)
    for i in range(n):
        sorted_abs[i] = abs_x[i]
    for i in range(1, n):
        key = sorted_abs[i]
        j = i - 1
        while j >= 0 and sorted_abs[j] > key:
            sorted_abs[j + 1] = sorted_abs[j]
            j -= 1
        sorted_abs[j + 1] = key

    best_i = 0
    best_mse = float("inf")

    for i in range(grid_len):
        p = grid[i]
        k = (p / 100.0) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            thr = float(sorted_abs[int(f)])
        else:
            d = k - f
            thr = float(sorted_abs[int(f)] * (1.0 - d) + sorted_abs[int(c)] * d)

        if thr <= 0.0:
            thr = 1e-8

        scale = thr / qmax
        sum_sq_err = 0.0

        for j in range(n):
            val = x[j]
            if val < -thr:
                clipped = -thr
            elif val > thr:
                clipped = thr
            else:
                clipped = val

            code = round(clipped / scale)
            if code < -qmax:
                code = -qmax
            elif code > qmax:
                code = qmax

            deq = code * scale
            diff = val - deq
            sum_sq_err += diff * diff

        mse = sum_sq_err / n

        if mse < best_mse:
            best_mse = mse
            best_i = i

    return best_i, best_mse
