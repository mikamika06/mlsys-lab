import numpy as np


def _qd_1d(x, bits):
    x = np.asarray(x, dtype=np.float64)
    qmax = (1 << bits) - 1
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    if xmax <= xmin:
        return x.copy()
    scale = (xmax - xmin) / qmax
    zero = round(-xmin / scale)
    zero = min(max(zero, 0), qmax)
    codes = np.clip(np.round(x / scale) + zero, 0, qmax)
    return (codes - zero) * scale


def _group_quant(x, axis, group_size, bits):
    x = np.asarray(x, dtype=np.float64)
    rows, cols = x.shape
    out = np.empty_like(x)
    if axis == 0:
        for c in range(cols):
            col = x[:, c]
            for s in range(0, rows, group_size):
                seg = col[s:s + group_size]
                out[s:s + group_size, c] = _qd_1d(seg, bits)
    else:
        for r in range(rows):
            row = x[r]
            for s in range(0, cols, group_size):
                seg = row[s:s + group_size]
                out[r, s:s + group_size] = _qd_1d(seg, bits)
    return out


def quantize_dequantize_kv(K, V, group_size, bits=4):
    K_hat = _group_quant(K, axis=0, group_size=group_size, bits=bits)
    V_hat = _group_quant(V, axis=1, group_size=group_size, bits=bits)
    return K_hat, V_hat
