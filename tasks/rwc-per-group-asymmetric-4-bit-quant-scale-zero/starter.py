import numpy as np


def quantize_group_affine_uint4(W, group_size):
    """Per-group asymmetric affine 4-bit quantization (GPTQ-style).

    Ravel W in row-major order, split into consecutive groups of
    `group_size` elements (last group may be shorter). Per group:
      scale = (max - min) / 15   (use 1.0 if the group is constant)
      zero  = clip(round(-min / scale), 0, 15)
      code  = clip(round(x / scale) + zero, 0, 15)

    Returns (codes, scale, zero):
      codes -- uint8 array, same shape as W, values in [0, 15]
      scale -- float64 array, one entry per group
      zero  -- float64 array, one entry per group
    """
    raise NotImplementedError('your code here')
