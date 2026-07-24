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
    amax = float(np.max(np.abs(x)))
    if amax == 0.0:
        return 0.0, 0.0

    scale = amax / fmax
    scaled = np.clip(x / scale, -fmax, fmax)

    grid = _GRIDS[fmt]
    flat = scaled.ravel()
    diffs = np.abs(flat[:, None] - grid[None, :])
    idx = np.argmin(diffs, axis=1)
    q = grid[idx].reshape(scaled.shape)

    dequant = q * scale
    err = float(np.max(np.abs(dequant - x)))
    return scale, err
