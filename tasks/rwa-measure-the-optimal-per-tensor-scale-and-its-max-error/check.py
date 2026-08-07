import numpy as np
from mlsys.scorers import rel_err


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


def _nearest(clipped, grid):
    flat = clipped.ravel()
    diffs = np.abs(flat[:, None] - grid[None, :])
    idx = np.argmin(diffs, axis=1)
    return grid[idx].reshape(clipped.shape)


def _oracle_scale_and_error(x, fmt):
    x_arr = np.asarray(x, dtype=np.float64)
    fmax = _FP8_MAX[fmt]
    amax = float(np.max(np.abs(x_arr)))
    if amax == 0.0:
        return 0.0, 0.0
    scale = amax / fmax
    scaled = np.clip(x_arr / scale, -fmax, fmax)
    q = _nearest(scaled, _GRIDS[fmt])
    dequant = q * scale
    err = float(np.max(np.abs(dequant - x_arr)))
    return scale, err


def _make_well_scaled(rng, shape):
    return rng.standard_normal(shape) * 4.0


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(31)

    worst_scale_err = 0.0
    worst_error_err = 0.0

    cases = []
    for _ in range(6):
        shape = tuple(int(v) for v in rng.integers(4, 20, size=2))
        scale_factor = float(rng.uniform(0.1, 500.0))
        x_arr = rng.standard_normal(shape) * scale_factor
        x_list = x_arr.tolist()
        fmt = "e4m3" if rng.random() < 0.5 else "e5m2"
        cases.append((x_arr, x_list, fmt))

    for x_arr, x_list, fmt in cases:
        ref_scale, ref_err = _oracle_scale_and_error(x_arr, fmt)
        try:
            got_scale, got_err = sol.optimal_scale_and_error(list(x_list), fmt)
            got_scale = float(got_scale)
            got_err = float(got_err)
        except Exception:
            return {"scale_rel_err": float("inf"), "error_rel_err": float("inf"), "order_ok": 0.0}

        worst_scale_err = max(worst_scale_err, rel_err(np.array([ref_scale]), np.array([got_scale])))
        worst_error_err = max(worst_error_err, rel_err(np.array([ref_err]), np.array([got_err])))

    order_ok = 1.0
    for _ in range(4):
        shape = tuple(int(v) for v in rng.integers(8, 24, size=2))
        x_arr = _make_well_scaled(rng, shape)
        x_list = x_arr.tolist()
        try:
            _, e4_err = sol.optimal_scale_and_error(list(x_list), "e4m3")
            _, e5_err = sol.optimal_scale_and_error(list(x_list), "e5m2")
        except Exception:
            order_ok = 0.0
            break
        if not (float(e4_err) < float(e5_err)):
            order_ok = 0.0
            break

    return {
        "scale_rel_err": worst_scale_err,
        "error_rel_err": worst_error_err,
        "order_ok": order_ok,
    }
