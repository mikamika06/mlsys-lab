import numpy as np


def nf4_levels() -> np.ndarray:
    """Return the 16 NF4 codebook levels (equal-probability-mass N(0,1)
    quantiles, normalized to [-1, 1], sorted ascending). See task.md."""
    raise NotImplementedError('your code here')


def quantize_4bit(x: np.ndarray, block_size: int = 64) -> tuple:
    """Blockwise absmax NF4 quantization. Returns (packed_uint8, absmax_float32).
    See task.md for the exact packing layout."""
    raise NotImplementedError('your code here')


def dequantize_4bit(packed: np.ndarray, absmax: np.ndarray, n: int, block_size: int = 64) -> np.ndarray:
    """Inverse of quantize_4bit. Returns a length-n float array."""
    raise NotImplementedError('your code here')
