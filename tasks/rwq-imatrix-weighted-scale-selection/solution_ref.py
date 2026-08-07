def imatrix_best_scale(x: list[float], w: list[float], scale_grid: list[float], qmin: int, qmax: int) -> int:
    """
    Sweep the candidate scale grid; for each scale, symmetric-quantize x
    (round(x/s), clipped to [qmin, qmax]), dequantize, and score with the
    imatrix-weighted squared error sum(w * (x - xhat)**2). Return the index
    of the grid entry with the smallest weighted error (first on ties).
    """
    best_i = 0
    best_err = float("inf")
    n = len(x)

    for i in range(len(scale_grid)):
        s = scale_grid[i]
        err = 0.0
        for j in range(n):
            val = x[j] / s
            if val >= 0.0:
                q = int(val + 0.5)
            else:
                q = int(val - 0.5)
            if q < qmin:
                q = qmin
            elif q > qmax:
                q = qmax
            xhat = q * s
            diff = x[j] - xhat
            err += w[j] * (diff * diff)

        if err < best_err:
            best_err = err
            best_i = i

    return best_i
