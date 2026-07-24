import numpy as np

from mlsys import scorers


def _decode_e4m3fn_one(code: int) -> float:
    sign = -1.0 if (code & 0x80) else 1.0
    e = (code >> 3) & 0x0F
    m = code & 0x07
    if e == 0:
        return sign * (m / 8.0) * (2.0 ** -6)
    if e == 15 and m == 7:
        return float("nan")
    return sign * (1.0 + m / 8.0) * (2.0 ** (e - 7))


def _grid_e4m3fn() -> np.ndarray:
    vals = set()
    for code in range(0, 128):
        if code == 0x7F:
            continue
        vals.add(_decode_e4m3fn_one(code))
    return np.array(sorted(vals), dtype=np.float64)


_GRID = _grid_e4m3fn()
_MAXV = float(_GRID[-1])


def _nearest_grid(values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(_GRID, values)
    idx = np.clip(idx, 1, len(_GRID) - 1)
    lo = _GRID[idx - 1]
    hi = _GRID[idx]
    choose_hi = (hi - values) < (values - lo)
    return np.where(choose_hi, hi, lo)


def _ref_per_channel(W: np.ndarray):
    W = np.asarray(W, dtype=np.float64)
    rows = W.shape[0]
    scales = np.empty(rows, dtype=np.float64)
    out = np.empty_like(W)
    for i in range(rows):
        row = W[i]
        amax = np.max(np.abs(row))
        scale = amax / _MAXV if amax > 0 else 1.0
        scales[i] = scale
        y = row / scale
        sign = np.sign(y)
        mag = np.clip(np.abs(y), 0.0, _MAXV)
        q = _nearest_grid(mag)
        out[i] = sign * q * scale
    return scales, out


def _scenarios():
    rng = np.random.default_rng(0)
    scenarios = []

    W1 = np.zeros((3, 5))
    W1[0] = 100.0
    W1[1, 0] = 0.01
    W1[1, 2] = -0.02
    scenarios.append(W1)

    for rows, cols, spread in [(4, 8, 100.0), (2, 4, 1000.0), (6, 16, 10.0)]:
        base = rng.normal(size=(rows, cols))
        per_row_scale = rng.uniform(0.01, spread, size=(rows, 1))
        W = base * per_row_scale
        scenarios.append(W)

    return scenarios


def grade(sol, fx) -> dict:
    worst = 0.0
    for W in _scenarios():
        scales_ref, out_ref = _ref_per_channel(W)
        try:
            scales_got, out_got = sol.per_channel_fp8_quant(W.copy())
        except Exception:
            return {"max_abs_err": float("inf")}

        try:
            scales_got = np.asarray(scales_got, dtype=np.float64)
            out_got = np.asarray(out_got, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if scales_got.shape != scales_ref.shape or out_got.shape != out_ref.shape:
            return {"max_abs_err": float("inf")}

        err = max(
            scorers.max_abs_err(scales_ref, scales_got),
            scorers.max_abs_err(out_ref, out_got),
        )
        if not np.isfinite(err):
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)

    return {"max_abs_err": worst}
