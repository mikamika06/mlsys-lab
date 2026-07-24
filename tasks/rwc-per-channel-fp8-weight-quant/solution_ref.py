import numpy as np


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


def per_channel_fp8_quant(W: np.ndarray):
    """Per-row (per-output-channel) E4M3 quantize/dequantize.

    Returns (scales, W_dequant): scales has shape (rows,), W_dequant has
    the same shape as W. Each row's scale is max(|row|)/448, or 1.0 for
    an all-zero row.
    """
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
