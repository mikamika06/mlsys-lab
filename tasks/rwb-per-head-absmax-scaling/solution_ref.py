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


def _nearest_grid(values):
    values = np.asarray(values, dtype=np.float64)
    shape = values.shape
    res_list = []
    for v in values.flat:
        idx = _binary_search_left(_GRID, v)
        if idx < 1:
            idx = 1
        elif idx > len(_GRID) - 1:
            idx = len(_GRID) - 1
        lo = _GRID[idx - 1]
        hi = _GRID[idx]
        if (hi - v) < (v - lo):
            res_list.append(hi)
        else:
            res_list.append(lo)
    return np.array(res_list, dtype=np.float64).reshape(shape)


def per_head_absmax_e4m3(k):
    k = np.asarray(k, dtype=np.float64)
    H = k.shape[0]
    out = np.empty_like(k)
    for h in range(H):
        blk = k[h]
        amax = 0.0
        for val in blk.flat:
            av = abs(val)
            if av > amax:
                amax = av
        scale = amax / _MAXV if amax > 0 else 1.0
        shape = blk.shape
        res_list = []
        for val in blk.flat:
            y_val = val / scale
            if y_val == 0:
                s_val = y_val
            else:
                s_val = 1.0 if y_val > 0 else -1.0
            abs_y = abs(y_val)
            mag_val = abs_y if abs_y <= _MAXV else _MAXV
            idx = _binary_search_left(_GRID, mag_val)
            if idx < 1:
                idx = 1
            elif idx > len(_GRID) - 1:
                idx = len(_GRID) - 1
            lo = _GRID[idx - 1]
            hi = _GRID[idx]
            if (hi - mag_val) < (mag_val - lo):
                q_val = hi
            else:
                q_val = lo
            res_list.append(s_val * q_val * scale)
        out[h] = np.array(res_list, dtype=np.float64).reshape(shape)
    return out
