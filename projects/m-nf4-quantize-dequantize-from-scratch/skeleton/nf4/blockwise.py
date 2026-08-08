import numpy as np


def quantize_blockwise(x: np.ndarray, codebook: np.ndarray, block_size: int = 64):
    """Quantize array into 4-bit indices (0-15) and per-block absmax scale factors."""
    raise NotImplementedError


def dequantize_blockwise(indices: np.ndarray, scales: np.ndarray, codebook: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Dequantize 4-bit indices and scale factors back to float representation."""
    raise NotImplementedError
