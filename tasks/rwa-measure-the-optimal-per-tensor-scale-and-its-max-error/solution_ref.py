import numpy as np


def _e4m3_grid():
    vals = set()
    for exp in range(16):
        for mant in range(8):
            if exp == 15 and mant == 7:
                continue
            if exp == 0:
                val = (2 ** -6) * (mant / 8.0)
            else:
                val = (2 ** (exp - 7)) * (1.0 + mant / 8.0)
            vals.add(val)
            vals.add(-val)
    return np.array(sorted(vals), dtype=np.float64)


def _e5m2_grid():
    vals = set()
    for exp in range(31):
        for mant in range(4):
            if exp == 0:
                val = (2 ** -14) * (mant / 4.0)
            else:
                val = (2 ** (exp - 15)) * (1.0 + mant / 4.0)
            vals.add(val)
            vals.add(-val)
    return np.array(sorted(vals), dtype=np.float64)


_FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
_GRIDS = {"e4m3": _e4m3_grid(), "e5m2": _e5m2_grid()}


def optimal_scale_and_error(x: np.ndarray, fmt: str) -> tuple[float, float]:
    """
    Compute the standard per-tensor calibration scale for FP8 format `fmt`
    ("e4m3" or "e5m2"):

        scale = amax(|x|) / FORMAT_MAX[fmt]

    which maps the tensor's peak magnitude exactly onto the format's largest
    representable value. Quantize x/scale to the nearest representable
    minifloat value (clamped at +-FORMAT_MAX), dequantize by multiplying
    back by `scale`, and return (scale, max_abs_dequant_error) where
    max_abs_dequant_error = max_i |dequant(x)_i - x_i|.

    If amax(|x|) == 0, returns (0.0, 0.0).
    """
    x = np.asarray(x, dtype=np.float64)
    fmax = _FP8_MAX[fmt]
    
    amax = 0.0
    for v in np.nditer(x):
        val = float(v)
        if val < 0.0:
            val = -val
        if val > amax:
            amax = val

    if amax == 0.0:
        return 0.0, 0.0

    scale = amax / fmax
    grid = _GRIDS[fmt]

    max_err = 0.0
    q = np.empty(x.shape, dtype=np.float64)
    dequant = np.empty(x.shape, dtype=np.float64)

    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        val = float(x[idx])

        v = val / scale
        if v > fmax:
            v = fmax
        elif v < -fmax:
            v = -fmax

        min_diff = float("inf")
        best_g = grid[0]
        for g in grid:
            diff = v - g
            if diff < 0.0:
                diff = -diff
            if diff < min_diff:
                min_diff = diff
                best_g = g

        q[idx] = best_g
        dq = best_g * scale
        dequant[idx] = dq

        err_val = dq - val
        if err_val < 0.0:
            err_val = -err_val
        if err_val > max_err:
            max_err = err_val

        it.iternext()

    return scale, max_err
