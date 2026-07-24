import numpy as np


def _qd_1d(x, bits=4):
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << bits) - 1
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        return x.copy()
    scale = (xmax - xmin) / qmax
    zero = round(-xmin / scale)
    zero = min(max(zero, 0), qmax)
    codes = np.clip(np.round(x / scale + zero), 0, qmax)
    return (codes - zero) * scale


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
