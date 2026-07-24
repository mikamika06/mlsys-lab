import numpy as np


def quantize_group_affine_uint4(W, group_size):
    """GPTQ-style per-group asymmetric affine 4-bit quantization.

    W is raveled in row-major order and split into consecutive groups of
    `group_size` elements (the last group may be shorter). For each group:

        scale = (max - min) / 15          (1.0 if the group is constant)
        zero  = clip(round(-min / scale), 0, 15)
        code  = clip(round(x / scale) + zero, 0, 15)   for each x in the group

    Returns (codes, scale, zero):
      codes -- uint8 array, same shape as W, values in [0, 15]
      scale -- float64 array, one entry per group
      zero  -- float64 array, one entry per group
    """
    shape = W.shape
    flat = np.asarray(W, dtype=np.float64).ravel()
    n = flat.size
    codes = np.empty(n, dtype=np.uint8)
    scales = []
    zeros = []

    for start in range(0, n, group_size):
        g = flat[start:start + group_size]
        gmax = float(g.max())
        gmin = float(g.min())
        span = gmax - gmin
        scale = 1.0 if span == 0.0 else span / 15.0
        zero = float(np.clip(np.rint(-gmin / scale), 0, 15))
        code = np.clip(np.rint(g / scale) + zero, 0, 15).astype(np.uint8)
        codes[start:start + len(g)] = code
        scales.append(scale)
        zeros.append(zero)

    return codes.reshape(shape), np.asarray(scales, dtype=np.float64), np.asarray(zeros, dtype=np.float64)
