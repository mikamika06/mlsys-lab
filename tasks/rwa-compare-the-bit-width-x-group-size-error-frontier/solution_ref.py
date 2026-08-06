import numpy as np


def _quant_dequant_1d(x, bits):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    qmax = (1 << bits) - 1
    xmin = float(x[0])
    xmax = float(x[0])
    for i in range(1, n):
        val = float(x[i])
        if val < xmin:
            xmin = val
        if val > xmax:
            xmax = val
    if xmax <= xmin:
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = float(x[i])
        return out
    scale = (xmax - xmin) / qmax
    zero = round(-xmin / scale)
    if zero < 0:
        zero = 0
    elif zero > qmax:
        zero = qmax
    res = np.empty(n, dtype=np.float64)
    for i in range(n):
        val = float(x[i]) / scale + zero
        rounded = round(val)
        if rounded < 0:
            code = 0
        elif rounded > qmax:
            code = qmax
        else:
            code = rounded
        res[i] = (code - zero) * scale
    return res


def _grouped_dequant(W, bits, group_size):
    W = np.asarray(W, dtype=np.float64)
    rows, cols = W.shape
    if group_size is None:
        flat = np.empty(rows * cols, dtype=np.float64)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                flat[idx] = W[r, c]
                idx += 1
        dequantized = _quant_dequant_1d(flat, bits)
        out = np.empty((rows, cols), dtype=np.float64)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                out[r, c] = dequantized[idx]
                idx += 1
        return out

    out = np.empty((rows, cols), dtype=np.float64)
    for r in range(rows):
        for start in range(0, cols, group_size):
            end = min(start + group_size, cols)
            seg_len = end - start
            seg = np.empty(seg_len, dtype=np.float64)
            for i in range(seg_len):
                seg[i] = W[r, start + i]
            dequant_seg = _quant_dequant_1d(seg, bits)
            for i in range(seg_len):
                out[r, start + i] = dequant_seg[i]
    return out


def bitwidth_group_mse_frontier(W, bit_options, group_size_options):
    W = np.asarray(W, dtype=np.float64)
    rows, cols = W.shape
    num_elements = rows * cols
    mse = np.zeros((len(bit_options), len(group_size_options)), dtype=np.float64)
    for bi, bits in enumerate(bit_options):
        for gi, g in enumerate(group_size_options):
            W_hat = _grouped_dequant(W, bits, g)
            total_sq_err = 0.0
            for r in range(rows):
                for c in range(cols):
                    diff = W_hat[r, c] - W[r, c]
                    total_sq_err += diff * diff
            mse[bi, gi] = total_sq_err / num_elements
    return mse
