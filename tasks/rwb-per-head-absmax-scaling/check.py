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


def _ref_per_head(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.float64)
    H = k.shape[0]
    out = np.empty_like(k)
    for h in range(H):
        blk = k[h]
        amax = np.max(np.abs(blk))
        scale = amax / _MAXV if amax > 0 else 1.0
        y = blk / scale
        sign = np.sign(y)
        mag = np.clip(np.abs(y), 0.0, _MAXV)
        q = _nearest_grid(mag)
        out[h] = sign * q * scale
    return out


def _scenarios():
    rng = np.random.default_rng(0)
    scenarios = []

    # hand-built: wildly different magnitudes across heads, plus all-zero head
    k1 = np.zeros((3, 4, 5))
    k1[0] = 100.0
    k1[1, 0, 0] = 0.01
    k1[1, 1, 2] = -0.02
    # k1[2] stays all zero
    scenarios.append(k1)

    # random tensors with per-head scale spread
    for heads, seq, head_dim, spread in [(4, 6, 8, 100.0), (2, 10, 4, 1000.0), (6, 3, 16, 10.0)]:
        base = rng.normal(size=(heads, seq, head_dim))
        per_head_scale = rng.uniform(0.01, spread, size=(heads, 1, 1))
        k = base * per_head_scale
        scenarios.append(k)

    return scenarios


def grade(sol, fx) -> dict:
    worst = 0.0
    for k in _scenarios():
        ref = _ref_per_head(k)
        try:
            got = sol.per_head_absmax_e4m3(k.copy())
        except Exception:
            return {"rel_err": float("inf")}

        try:
            got = np.asarray(got, dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}

        if got.shape != ref.shape:
            return {"rel_err": float("inf")}

        err = scorers.rel_err(ref, got)
        if not np.isfinite(err):
            return {"rel_err": float("inf")}
        worst = max(worst, err)

    return {"rel_err": worst}
