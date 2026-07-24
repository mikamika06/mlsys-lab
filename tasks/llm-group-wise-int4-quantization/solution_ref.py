import numpy as np


def quantize_groupwise_int4(W, group_size):
    shape = W.shape
    flat = np.asarray(W, dtype=np.float64).ravel()
    q = np.empty(flat.shape, dtype=np.int8)
    scales = []

    for start in range(0, len(flat), group_size):
        group = flat[start:start + group_size]
        max_abs = np.max(np.abs(group))
        scale = 1.0 if max_abs == 0 else max_abs / 7.0
        scales.append(scale)
        values = np.rint(group / scale)
        values = np.clip(values, -8, 7)
        q[start:start + len(group)] = values.astype(np.int8)

    return q, np.asarray(scales, dtype=np.float64), shape
