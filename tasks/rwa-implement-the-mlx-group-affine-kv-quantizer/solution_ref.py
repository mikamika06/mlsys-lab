import numpy as np


def quantize_kv_group_affine(kv, kv_bits, kv_group_size):
    x = np.asarray(kv, dtype=np.float64)
    groups = x.shape[-1] // kv_group_size
    g = x.reshape(x.shape[:-1] + (groups, kv_group_size))

    qmax = (1 << kv_bits) - 1
    xmin = np.min(g, axis=-1, keepdims=True)
    xmax = np.max(g, axis=-1, keepdims=True)

    scale = (xmax - xmin) / qmax
    scale = np.where(scale == 0, 1.0, scale)

    zero = np.round(-xmin / scale)
    zero = np.clip(zero, 0, qmax)

    q = np.round(g / scale + zero)
    q = np.clip(q, 0, qmax).astype(np.int32)

    return q, scale, zero
