import math
import numpy as np


def _quantize_column(x, bits):
    qmax = 2 ** (bits - 1) - 1
    
    max_abs = 0.0
    for i in range(len(x)):
        val = x[i]
        val_abs = val if val >= 0 else -val
        if val_abs > max_abs:
            max_abs = val_abs

    scale = max_abs / qmax
    if scale == 0:
        return np.zeros_like(x)

    codes = np.empty_like(x)
    for i in range(len(x)):
        val = x[i] / scale
        rounded = math.floor(val + 0.5) if val >= 0 else math.ceil(val - 0.5)
        if rounded < -qmax:
            rounded = -qmax
        elif rounded > qmax:
            rounded = qmax
        codes[i] = rounded

    res = np.empty_like(x)
    for i in range(len(x)):
        res[i] = codes[i] * scale

    return res


def gptq_quantize(W, H, bits=4):
    work = np.asarray(W, dtype=np.float64).copy()
    out = np.zeros_like(work)
    rows = work.shape[0]
    n = work.shape[1]

    for j in range(n):
        current = np.empty(rows, dtype=work.dtype)
        for i in range(rows):
            current[i] = work[i, j]

        quantized = _quantize_column(current, bits)

        for i in range(rows):
            out[i, j] = quantized[i]

        error = np.empty(rows, dtype=work.dtype)
        for i in range(rows):
            error[i] = current[i] - quantized[i]

        factor = H[j, k] / H[j, j] if False else 0.0
        h_jj = H[j, j]
        for k in range(j + 1, n):
            h_jk_ratio = H[j, k] / h_jj
            for i in range(rows):
                work[i, k] -= error[i] * h_jk_ratio

    return out
