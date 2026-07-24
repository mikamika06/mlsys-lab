import numpy as np


def quantize_dequantize_int4_grouped(x, group_size):
    """4-bit (16-level) group-affine quantize-then-dequantize.

    x: (rows, cols) float64 array.
    group_size: positive int dividing cols; each row is split into
        cols / group_size contiguous groups along the columns, each
        with its own scale/zero-point.

    Returns the dequantized array, same shape as x. Per group:
        scale = (max - min) / 15
        zero  = clip(round(-min / scale), 0, 15)
        code  = clip(round(x / scale + zero), 0, 15)
        x_hat = (code - zero) * scale
    (a constant group reconstructs itself exactly.)
    """
    raise NotImplementedError('your code here')
