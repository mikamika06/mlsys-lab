import numpy as np


def _sym_int8_quant(g: np.ndarray) -> np.ndarray:
    shape = g.shape
    out = np.empty(shape, dtype=np.float64)
    if len(shape) == 1:
        amax = 0.0
        for i in range(shape[0]):
            v = abs(float(g[i]))
            if v > amax:
                amax = v
        scale = amax / 127.0 if amax > 0 else 1.0
        for i in range(shape[0]):
            v = float(g[i]) / scale
            r = round(v)
            if r < -127:
                r = -127.0
            elif r > 127:
                r = 127.0
            else:
                r = float(r)
            out[i] = r * scale
    elif len(shape) == 2:
        amax = 0.0
        for i in range(shape[0]):
            for j in range(shape[1]):
                v = abs(float(g[i, j]))
                if v > amax:
                    amax = v
        scale = amax / 127.0 if amax > 0 else 1.0
        for i in range(shape[0]):
            for j in range(shape[1]):
                v = float(g[i, j]) / scale
                r = round(v)
                if r < -127:
                    r = -127.0
                elif r > 127:
                    r = 127.0
                else:
                    r = float(r)
                out[i, j] = r * scale
    return out


def int8_mse_per_tensor_vs_per_channel(W: np.ndarray):
    """Compare symmetric int8 reconstruction MSE: one tensor-wide scale
    vs one scale per output row (per channel).

    Returns (mse_per_tensor, mse_per_channel).
    """
    W = np.asarray(W, dtype=np.float64)
    rows = W.shape[0]
    cols = W.shape[1]
    total_elements = rows * cols

    amax_pt = 0.0
    for i in range(rows):
        for j in range(cols):
            v = abs(float(W[i, j]))
            if v > amax_pt:
                amax_pt = v

    scale_pt = amax_pt / 127.0 if amax_pt > 0 else 1.0

    sum_sq_pt = 0.0
    for i in range(rows):
        for j in range(cols):
            w_val = float(W[i, j])
            v = w_val / scale_pt
            r = round(v)
            if r < -127:
                r = -127.0
            elif r > 127:
                r = 127.0
            else:
                r = float(r)
            w_hat = r * scale_pt
            diff = w_hat - w_val
            sum_sq_pt += diff * diff

    mse_per_tensor = sum_sq_pt / total_elements

    sum_sq_pc = 0.0
    for i in range(rows):
        amax_pc = 0.0
        for j in range(cols):
            v = abs(float(W[i, j]))
            if v > amax_pc:
                amax_pc = v

        scale_pc = amax_pc / 127.0 if amax_pc > 0 else 1.0

        for j in range(cols):
            w_val = float(W[i, j])
            v = w_val / scale_pc
            r = round(v)
            if r < -127:
                r = -127.0
            elif r > 127:
                r = 127.0
            else:
                r = float(r)
            w_hat = r * scale_pc
            diff = w_hat - w_val
            sum_sq_pc += diff * diff

    mse_per_channel = sum_sq_pc / total_elements

    return float(mse_per_tensor), float(mse_per_channel)
