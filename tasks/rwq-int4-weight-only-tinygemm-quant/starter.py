import numpy as np


def tinygemm_int4_quantize(W: np.ndarray, group_size: int = 128):
    """Group-wise asymmetric uint4 weight-only quantization (tinygemm layout).

    W: shape (rows, cols), cols an exact multiple of group_size.

    Returns (codes, scale, zero_point, dequantized):
      codes: uint8 array, shape (rows, cols), values in [0, 15].
      scale: float64 array, shape (rows, cols // group_size).
      zero_point: float64 array, shape (rows, cols // group_size), float-domain.
      dequantized: float64 array, shape (rows, cols).
    """
    raise NotImplementedError('your code here')
