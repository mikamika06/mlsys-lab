def _decode_e4m3fn_one(code: int) -> float:
    sign = -1.0 if (code & 0x80) else 1.0
    e = (code >> 3) & 0x0F
    m = code & 0x07
    if e == 0:
        return sign * (m / 8.0) * (2.0**-6)
    if e == 15 and m == 7:
        return float("nan")
    return sign * (1.0 + m / 8.0) * (2.0 ** (e - 7))


def _grid_e4m3fn() -> list[float]:
    vals = set()
    for code in range(0, 128):
        if code == 0x7F:
            continue
        vals.add(_decode_e4m3fn_one(code))
    return sorted(vals)


_GRID = _grid_e4m3fn()
_MAXV = float(_GRID[-1])


def per_channel_fp8_quant(
    W: list[list[float]],
) -> tuple[list[float], list[list[float]]]:
    """Per-row (per-output-channel) E4M3 quantize/dequantize.

    Returns (scales, W_dequant): scales has length rows, W_dequant has
    the same dimensions as W. Each row's scale is max(|row|)/448, or 1.0 for
    an all-zero row.
    """
    rows = len(W)
    cols = len(W[0]) if rows > 0 else 0
    scales = [0.0] * rows
    out = [[0.0] * cols for _ in range(rows)]
    grid_len = len(_GRID)

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
            high = grid_len
            while low < high:
                mid = (low + high) // 2
                if _GRID[mid] < mag:
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
            if (hi - mag) < (mag - lo):
                q = hi
            else:
                q = lo
            row_out[j] = sign * q * scale
    return scales, out
