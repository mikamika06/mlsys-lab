import numpy as np


def quantize_rtn(W, bits):
    """Round-to-nearest symmetric per-output-row quantize + dequantize.

    W: (d_out, d_in) float array. Returns a float array of the same shape
    holding the de-quantized weights.
    """
    raise NotImplementedError('your code here')


def quantize_gptq(W, X, bits, damp=0.01):
    """GPTQ (error-compensated) quantize + dequantize of W using calibration X.

    W: (d_out, d_in), X: (n_cal, d_in). Returns the de-quantized weights,
    same shape and dtype family as W.
    """
    raise NotImplementedError('your code here')
