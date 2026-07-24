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
    clipped = np.clip(x, -fmax, fmax)
    flat = clipped.ravel()
    diffs = np.abs(flat[:, None] - grid[None, :])
    idx = np.argmin(diffs, axis=1)
    return grid[idx].reshape(x.shape)


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
    e4m3_max_abs_err = float(np.max(np.abs(e4 - x)))
    e5m2_max_abs_err = float(np.max(np.abs(e5 - x)))
    return e4m3_max_abs_err, e5m2_max_abs_err
