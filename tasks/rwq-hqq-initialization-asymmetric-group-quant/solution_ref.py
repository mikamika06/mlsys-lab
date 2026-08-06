import numpy as np


def hqq_init(W, group_size=64, nbits=4):
    """HQQ-style asymmetric group quantization initializer.

    This is the closed-form starting point HQQ uses before its zero-point
    refinement loop kicks in: a plain per-group affine (min/max) quantizer.

    `W` is raveled in row-major order and split into consecutive groups of
    `group_size` elements (the last group may be shorter). For each group:

        scale = (max(g) - min(g)) / (2**nbits - 1)   (1.0 if the group is constant)
        zero  = round(-min(g) / scale)
        code  = clip(round(g / scale) + zero, 0, 2**nbits - 1)

    Parameters
    ----------
    W : np.ndarray
        Weight tensor of any shape.
    group_size : int
        Number of consecutive (raveled) elements sharing one (scale, zero).
    nbits : int
        Bit width of the quantization codes.

    Returns
    -------
    W_q : np.ndarray, uint8, same shape as W
        Quantization codes in [0, 2**nbits - 1].
    scale : np.ndarray, float64, shape (n_groups,)
    zero : np.ndarray, float64, shape (n_groups,)
    dequant : np.ndarray, float64, same shape as W
        Reconstruction: (W_q - zero) * scale, broadcast per group.
    """
    shape = W.shape
    flat = np.asarray(W, dtype=np.float64).ravel()
    n = flat.size
    qmax = 2 ** nbits - 1

    codes = np.empty(n, dtype=np.uint8)
    dequant = np.empty(n, dtype=np.float64)
    scales = []
    zeros = []

    for start in range(0, n, group_size):
        g = flat[start:start + group_size]
        gmax = g[0]
        gmin = g[0]
        for i in range(1, len(g)):
            val = g[i]
            if val > gmax:
                gmax = val
            if val < gmin:
                gmin = val
        gmax = float(gmax)
        gmin = float(gmin)
        span = gmax - gmin
        scale = 1.0 if span == 0.0 else span / qmax
        zero = float(round(-gmin / scale))

        for i in range(len(g)):
            val = g[i]
            r = round((val / scale) + zero)
            if r < 0:
                c = 0
            elif r > qmax:
                c = qmax
            else:
                c = r
            idx = start + i
            codes[idx] = c
            dequant[idx] = (float(c) - zero) * scale

        scales.append(scale)
        zeros.append(zero)

    return (
        codes.reshape(shape),
        np.asarray(scales, dtype=np.float64),
        np.asarray(zeros, dtype=np.float64),
        dequant.reshape(shape),
    )
