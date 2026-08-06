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
    flat = values.ravel()
    res = np.empty_like(flat)
    grid_len = len(_GRID)
    for i in range(len(flat)):
        val = flat[i]
        low = 0
        high = grid_len
        while low < high:
            mid = (low + high) // 2
            if _GRID[mid] < val:
                low = mid + 1
            else:
                high = mid
        idx = low
        if idx < 1:
            idx = 1
        elif idx >= grid_len:
            idx = grid_len - 1
        lo = _GRID[idx - 1]
        hi = _GRID[idx]
        if (hi - val) < (val - lo):
            res[i] = hi
        else:
            res[i] = lo
    return res.reshape(values.shape)


def per_channel_fp8_quant(W: np.ndarray):
    """Per-row (per-output-channel) E4M3 quantize/dequantize.

    Returns (scales, W_dequant): scales has shape (rows,), W_dequant has
    the same shape as W. Each row's scale is max(|row|)/448, or 1.0 for
    an all-zero row.
    """
    W = np.asarray(W, dtype=np.float64)
    rows = W.shape[0]
    cols = W.shape[1] if W.ndim > 1 else 1
    scales = np.empty(rows, dtype=np.float64)
    out = np.empty_like(W)
    for i in range(rows):
        row = W[i]
        amax = 0.0
        for j in range(cols):
            val = row[j]
            if val < 0.0:
                val = -val
            if val > amax:
                amax = val
        scale = amax / _MAXV if amax > 0.0 else 1.0
        scales[i] = scale
        
        row_out = out[i]
        for j in range(cols):
            val = row[j] / scale
            if val < 0.0:
                sign = -1.0
                mag = -val
            else:
                sign = 1.0
                mag = val
            if mag > _MAXV:
                mag = _MAXV
            
            low = 0
            high = len(_GRID)
            while low < high:
                mid = (low + high) // 2
                if _GRID[mid] < mag:
                    low = mid + 1
                else:
                    high = mid
            idx = low
            if idx < 1:
                idx = 1
            elif idx >= len(_GRID):
                idx = len(_GRID) - 1
            lo = _GRID[idx - 1]
            hi = _GRID[idx]
            if (hi - mag) < (mag - lo):
                q = hi
            else:
                q = lo
            row_out[j] = sign * q * scale
    return scales, out
