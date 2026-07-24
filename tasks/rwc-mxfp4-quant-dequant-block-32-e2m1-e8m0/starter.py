import numpy as np


def mxfp4_quant_dequant(weights):
    """
    weights: (B, 32) array -- B independent blocks of 32 values.

    For each block, compute the shared exponent
        e = max(0, ceil(log2(max(|block|) / 6))),
    snap every element of block / 2^e to the nearest signed E2M1 grid
    value in {0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6} to get that
    element's code, then dequantize as code * 2^e.

    Returns (codes, dequantized), both (B, 32) float64 arrays.
    """
    raise NotImplementedError('your code here')
