import numpy as np


def quantize_blockwise(tensor, codebook, block_size=64):
    """
    Quantize a 1D float32 tensor into 4-bit indices using blockwise quantization.

    Args:
        tensor: 1D numpy array of shape (N,), float32. N is a multiple of block_size.
        codebook: 1D numpy array of shape (16,), sorted.
        block_size: int, default 64.

    Returns:
        quantized: 1D numpy array of shape (N // 2,), dtype uint8.
                   The first element of a pair is in the high nibble (bits 4-7),
                   and the second element is in the low nibble (bits 0-3).
        absmax: 1D numpy array of shape (N // block_size,), float32.
    """
    raise NotImplementedError()


def dequantize_blockwise(quantized, absmax, codebook, block_size=64):
    """
    Dequantize a 1D uint8 tensor back to float32.
    """
    raise NotImplementedError()
