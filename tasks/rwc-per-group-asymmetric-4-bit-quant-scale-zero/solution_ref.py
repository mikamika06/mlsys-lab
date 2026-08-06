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
        end = start + group_size
        if end > n:
            end = n
        gmin = float('inf')
        gmax = float('-inf')
        for i in range(start, end):
            val = flat[i]
            if val < gmin:
                gmin = val
            if val > gmax:
                gmax = val
        span = gmax - gmin
        scale = 1.0 if span == 0.0 else span / 15.0
        raw_zero = round(-gmin / scale)
        if raw_zero < 0.0:
            zero = 0.0
        elif raw_zero > 15.0:
            zero = 15.0
        else:
            zero = float(raw_zero)
        scales.append(scale)
        zeros.append(zero)
        for i in range(start, end):
            raw_code = round(flat[i] / scale) + zero
            if raw_code < 0.0:
                code = 0
            elif raw_code > 15.0:
                code = 15
            else:
                code = int(raw_code)
            codes[i] = code

    return codes.reshape(shape), np.asarray(scales, dtype=np.float64), np.asarray(zeros, dtype=np.float64)
