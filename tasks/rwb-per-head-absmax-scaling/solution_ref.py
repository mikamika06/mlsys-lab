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
    return sorted(vals)


_GRID = _grid_e4m3fn()
_MAXV = float(_GRID[-1])


def _binary_search_left(arr, x):
    low = 0
    high = len(arr)
    while low < high:
        mid = (low + high) // 2
        if arr[mid] < x:
            low = mid + 1
        else:
            high = mid
    return low


def _nearest_grid_val(v):
    idx = _binary_search_left(_GRID, v)
    if idx < 1:
        idx = 1
    elif idx > len(_GRID) - 1:
        idx = len(_GRID) - 1
    lo = _GRID[idx - 1]
    hi = _GRID[idx]
    if (hi - v) < (v - lo):
        return hi
    else:
        return lo


def per_head_absmax_e4m3(k: list[list[list[float]]]) -> list[list[list[float]]]:
    H = len(k)
    out = []
    for h in range(H):
        blk = k[h]
        amax = 0.0
        for row in blk:
            for val in row:
                av = abs(val)
                if av > amax:
                    amax = av
        scale = amax / _MAXV if amax > 0 else 1.0
        scaled_blk = []
        for row in blk:
            new_row = []
            for val in row:
                y_val = val / scale
                if y_val == 0:
                    s_val = y_val
                else:
                    s_val = 1.0 if y_val > 0 else -1.0
                abs_y = abs(y_val)
                mag_val = abs_y if abs_y <= _MAXV else _MAXV
                q_val = _nearest_grid_val(mag_val)
                new_row.append(s_val * q_val * scale)
            scaled_blk.append(new_row)
        out.append(scaled_blk)
    return out
