import numpy as np


def hqq_init(W, group_size=64, nbits=4):
    """HQQ-style asymmetric group quantization initializer.

    Ravel W in row-major order, split into consecutive groups of
    `group_size` elements (last group may be shorter). Per group:

        scale = (max(g) - min(g)) / (2**nbits - 1)   (1.0 if constant)
        zero  = round(-min(g) / scale)
        code  = clip(round(g / scale) + zero, 0, 2**nbits - 1)

    Returns (W_q, scale, zero, dequant):
      W_q     -- uint8 array, same shape as W, codes in [0, 2**nbits - 1]
      scale   -- float64 array, one entry per group
      zero    -- float64 array, one entry per group
      dequant -- float64 array, same shape as W, (W_q - zero) * scale
    """
    raise NotImplementedError('your code here')
