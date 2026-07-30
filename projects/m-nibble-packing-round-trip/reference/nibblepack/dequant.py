import numpy as np

from .pack import unpack_nibbles

CODEBOOK = np.array([
    -1.0, -0.84, -0.68, -0.52,
    -0.36, -0.20, -0.04, 0.0,
    0.04, 0.12, 0.20, 0.28,
    0.40, 0.55, 0.75, 1.0,
], dtype=np.float64)


def dequantize_block(packed, n, absmax, codebook=CODEBOOK):
    codes = unpack_nibbles(packed, n)
    codebook = np.asarray(codebook, dtype=np.float64)
    return codebook[codes] * float(absmax)
