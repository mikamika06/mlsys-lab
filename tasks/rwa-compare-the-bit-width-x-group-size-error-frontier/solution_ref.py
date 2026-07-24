import numpy as np


def _quant_dequant_1d(x, bits):
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


def _grouped_dequant(W, bits, group_size):
    W = np.asarray(W, dtype=np.float64)
    rows, cols = W.shape
    if group_size is None:
        flat = W.reshape(-1)
        return _quant_dequant_1d(flat, bits).reshape(rows, cols)

    out = np.empty_like(W)
    for r in range(rows):
        row = W[r]
        for start in range(0, cols, group_size):
            seg = row[start:start + group_size]
            out[r, start:start + group_size] = _quant_dequant_1d(seg, bits)
    return out


def bitwidth_group_mse_frontier(W, bit_options, group_size_options):
    W = np.asarray(W, dtype=np.float64)
    mse = np.zeros((len(bit_options), len(group_size_options)), dtype=np.float64)
    for bi, bits in enumerate(bit_options):
        for gi, g in enumerate(group_size_options):
            W_hat = _grouped_dequant(W, bits, g)
            mse[bi, gi] = float(np.mean((W_hat - W) ** 2))
    return mse
