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


def _round_trip_err(v, fmt):
    grid = _GRIDS[fmt]
    fmax = _FP8_MAX[fmt]
    if v < -fmax:
        vc = -fmax
    elif v > fmax:
        vc = fmax
    else:
        vc = v
    
    best_idx = 0
    min_abs_diff = abs(grid[0] - vc)
    for i in range(1, len(grid)):
        diff = abs(grid[i] - vc)
        if diff < min_abs_diff:
            min_abs_diff = diff
            best_idx = i
            
    return abs(grid[best_idx] - v)


def classify_better_format(values: np.ndarray) -> np.ndarray:
    """
    For each scalar in `values`, round-trip it (quantize to the nearest
    representable value, clamped/saturated at the format's finite max
    magnitude -- no rescaling) through both E4M3 and E5M2, and label which
    format gives the SMALLER absolute round-trip error:

        0 -> E4M3 wins (or exact tie)
        1 -> E5M2 wins

    Returns an int array of the same shape as `values`.
    """
    values = np.asarray(values, dtype=np.float64)
    flat = values.ravel()
    out = np.empty(flat.shape, dtype=np.int64)
    for i, v in enumerate(flat):
        e4 = _round_trip_err(v, "e4m3")
        e5 = _round_trip_err(v, "e5m2")
        out[i] = 0 if e4 <= e5 else 1
    
    res = np.empty(values.shape, dtype=np.int64)
    flat_res = res.ravel()
    for i in range(len(out)):
        flat_res[i] = out[i]
    return res
