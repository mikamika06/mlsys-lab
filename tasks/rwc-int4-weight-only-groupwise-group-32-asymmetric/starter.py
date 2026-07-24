import numpy as np


def int4_groupwise_asymmetric(W: np.ndarray, X: np.ndarray, group_size: int):
    """
    INT4 weight-only, groupwise, asymmetric quantization: per contiguous
    group of `group_size` input-dim weights, compute (scale, zero) via
    min-max, quantize to uint8 codes in [0, 15], dequantize, and matmul
    with X. Returns (codes, scales, zeros, output). See task.md.
    """
    raise NotImplementedError('your code here')
