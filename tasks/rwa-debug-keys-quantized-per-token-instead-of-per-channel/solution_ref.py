import numpy as np


def quantize_keys_per_channel(K, bits=4):
    K = np.asarray(K, dtype=np.float64)
    levels = (2.0 ** (bits - 1)) - 1.0
    rows = K.shape[0]
    cols = K.shape[1]
    scale = [0.0] * cols
    for j in range(cols):
        max_val = 0.0
        for i in range(rows):
            val = K[i, j]
            if val < 0.0:
                val = -val
            if val > max_val:
                max_val = val
        s = max_val / levels
        if s == 0.0:
            s = 1.0
        scale[j] = s
    out = np.zeros((rows, cols), dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            s = scale[j]
            q = round(K[i, j] / s)
            out[i, j] = q * s
    return out
