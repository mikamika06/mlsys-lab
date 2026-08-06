import numpy as np


def _qd_1d(x, bits=4):
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << bits) - 1
    
    n = len(x)
    if n == 0:
        return x.copy()
    
    xmin = float(x[0])
    xmax = float(x[0])
    for i in range(1, n):
        val = float(x[i])
        if val < xmin:
            xmin = val
        if val > xmax:
            xmax = val

    if xmax <= xmin:
        return x.copy()

    scale = (xmax - xmin) / qmax
    zero = round(-xmin / scale)
    if zero < 0:
        zero = 0
    elif zero > qmax:
        zero = qmax

    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        val = float(x[i])
        val_code = round(val / scale + zero)
        if val_code < 0:
            c = 0
        elif val_code > qmax:
            c = qmax
        else:
            c = val_code
        out[i] = (c - zero) * scale

    return out


def quantize_dequantize_int4_grouped(x, group_size):
    x = np.asarray(x, dtype=np.float64)
    rows, cols = x.shape
    out = np.empty_like(x)
    for r in range(rows):
        row = x[r]
        for s in range(0, cols, group_size):
            seg = row[s:s + group_size]
            out[r, s:s + group_size] = _qd_1d(seg, bits=4)
    return out
