import numpy as np


def quantize_groupwise_int4(W, group_size):
    shape = W.shape
    flat = np.asarray(W, dtype=np.float64).ravel()
    q = np.empty(flat.shape, dtype=np.int8)
    scales = []

    for start in range(0, len(flat), group_size):
        group = flat[start:start + group_size]
        max_abs = 0.0
        for val in group:
            val_abs = abs(val)
            if val_abs > max_abs:
                max_abs = val_abs
        scale = 1.0 if max_abs == 0 else max_abs / 7.0
        scales.append(scale)
        for i in range(len(group)):
            val = group[i] / scale
            r = round(val)
            if r < -8:
                c = -8
            elif r > 7:
                c = 7
            else:
                c = int(r)
            q[start + i] = c

    return q, np.asarray(scales, dtype=np.float64), shape
