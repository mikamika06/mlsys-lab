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


def _quant_dequant_raw(x, fmt):
    grid = _GRIDS[fmt]
    fmax = _FP8_MAX[fmt]
    x = np.asarray(x, dtype=np.float64)
    out = []
    for val in x.flat:
        if val > fmax:
            v = fmax
        elif val < -fmax:
            v = -fmax
        else:
            v = float(val)
        best_diff = float("inf")
        best_g = grid[0]
        for g in grid:
            diff = v - g
            if diff < 0:
                diff = -diff
            if diff < best_diff:
                best_diff = diff
                best_g = g
        out.append(best_g)
    return np.array(out, dtype=np.float64).reshape(x.shape)


def fp8_format_errors(x: np.ndarray) -> tuple[float, float]:
    """
    Quantize-then-dequantize `x` through the two OCP 8-bit float formats,
    E4M3 and E5M2, with NO rescaling (raw values, saturating at each
    format's finite max magnitude). Return the maximum absolute
    reconstruction error for each format.
    """
    x = np.asarray(x, dtype=np.float64)
    e4 = _quant_dequant_raw(x, "e4m3")
    e5 = _quant_dequant_raw(x, "e5m2")
    e4m3_max_abs_err = 0.0
    for q, orig in zip(e4.flat, x.flat):
        err = q - orig
        if err < 0:
            err = -err
        if err > e4m3_max_abs_err:
            e4m3_max_abs_err = err
    e5m2_max_abs_err = 0.0
    for q, orig in zip(e5.flat, x.flat):
        err = q - orig
        if err < 0:
            err = -err
        if err > e5m2_max_abs_err:
            e5m2_max_abs_err = err
    return float(e4m3_max_abs_err), float(e5m2_max_abs_err)
