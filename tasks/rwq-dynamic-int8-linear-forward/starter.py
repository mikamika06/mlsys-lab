import numpy as np


def int8_linear_forward(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Dynamic int8 Linear forward: per-channel symmetric int8 weight quant +
    per-tensor dynamic asymmetric uint8 activation quant, integer matmul
    with zero-point correction, then dequantize. See task.md.
    """
    raise NotImplementedError('your code here')
