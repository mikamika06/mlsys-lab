import math
import numpy as np


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p):
    low, high = -10.0, 10.0
    for _ in range(100):
        mid = (low + high) / 2.0
        if norm_cdf(mid) < p:
            low = mid
        else:
            high = mid
    return mid


def build_nf4_codebook():
    """
    Returns a numpy array of shape (16,) containing the NF4 codebook.
    """
    raise NotImplementedError


def quantize_tensor(w, codebook, block_size):
    """
    w: 1D numpy array. length is perfectly divisible by block_size.
    codebook: 1D numpy array of shape (16,).
    block_size: int.
    Returns:
        indices: 2D numpy array of shape (num_blocks, block_size) of dtype int.
        absmaxes: 1D numpy array of shape (num_blocks,) containing the absmax of each block.
    """
    raise NotImplementedError


def dequantize_tensor(indices, absmaxes, codebook):
    """
    Reconstructs the 1D tensor from blockwise indices and absmaxes.
    Returns a 1D numpy array.
    """
    raise NotImplementedError
