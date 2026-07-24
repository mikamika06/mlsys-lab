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
    vc = np.clip(v, -fmax, fmax)
    idx = int(np.argmin(np.abs(grid - vc)))
    return abs(grid[idx] - v)


def _oracle_labels(values):
    labels = np.empty(values.shape, dtype=np.int64)
    flat = values.ravel()
    out = np.empty(flat.shape, dtype=np.int64)
    for i, v in enumerate(flat):
        e4 = _round_trip_err(v, "e4m3")
        e5 = _round_trip_err(v, "e5m2")
        out[i] = 0 if e4 <= e5 else 1  # tie -> E4M3
    return out.reshape(values.shape)


def grade(sol, fx) -> dict:
    values = np.asarray(fx["values"], dtype=np.float64)
    ref = _oracle_labels(values)

    try:
        got = np.asarray(sol.classify_better_format(values.copy()))
    except Exception:
        return {"exact_match": 0.0}

    if got.shape != ref.shape:
        return {"exact_match": 0.0}

    try:
        got_int = got.astype(np.int64)
    except (TypeError, ValueError):
        return {"exact_match": 0.0}

    ok = 1.0 if np.array_equal(got_int, ref) else 0.0
    return {"exact_match": ok}
