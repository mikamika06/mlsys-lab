import numpy as np


def _decode_e4m3fn_one(code):
    sign = -1.0 if (code & 0x80) else 1.0
    e = (code >> 3) & 0x0F
    m = code & 0x07
    if e == 0:
        return sign * (m / 8.0) * (2.0 ** -6)
    if e == 15 and m == 7:
        return float("nan")
    return sign * (1.0 + m / 8.0) * (2.0 ** (e - 7))


def _grid_e4m3fn():
    vals = set()
    for code in range(0, 128):
        if code == 0x7F:
            continue
        vals.add(_decode_e4m3fn_one(code))
    return np.array(sorted(vals), dtype=np.float64)


_GRID = _grid_e4m3fn()
_MAXV = float(_GRID[-1])


def _nearest_grid(values):
    idx = np.searchsorted(_GRID, values)
    idx = np.clip(idx, 1, len(_GRID) - 1)
    lo = _GRID[idx - 1]
    hi = _GRID[idx]
    choose_hi = (hi - values) < (values - lo)
    return np.where(choose_hi, hi, lo)


def per_head_absmax_e4m3(k):
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
