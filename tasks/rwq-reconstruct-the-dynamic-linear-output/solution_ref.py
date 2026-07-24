import numpy as np


def dequant_linear_output(W_int8, w_scales, x_uint8, x_scale, x_zp):
    """Dequantize and compute int8 linear layer output."""
    W = W_int8.astype(np.float64)
    x_signed = x_uint8.astype(np.float64) - int(x_zp)
    acc = W @ x_signed
    y = acc * w_scales.astype(np.float64) * x_scale
    return y.astype(np.float32)
