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


def _group_quant_rows(W, group_size, bits):
    rows, cols = W.shape
    out = np.empty_like(W)
    for r in range(rows):
        row = W[r]
        for c0 in range(0, cols, group_size):
            seg = row[c0:c0 + group_size]
            out[r, c0:c0 + group_size] = _qd_1d(seg, bits)
    return out


def awq_scale_and_quantize(W, X, s, group_size, bits=4):
    Wp = W * s[None, :]
    Xp = X / s[None, :]

    Y_identity = Xp @ Wp.T

    W_hat = _group_quant_rows(Wp, group_size, bits)
    Y_quant = Xp @ W_hat.T

    return Y_identity, Y_quant
