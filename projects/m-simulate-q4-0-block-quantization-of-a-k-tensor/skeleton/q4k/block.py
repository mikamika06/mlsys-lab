import numpy as np


def quantize_block_q4_0(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Quantize a 32-element float32 1D array into q4_0 scale and packed uint8 nibbles."""
    raise NotImplementedError


def dequantize_block_q4_0(d: float, qs: np.ndarray) -> np.ndarray:
    """Dequantize scale d and 16 packed uint8 nibbles into a 32-element float32 array."""
    raise NotImplementedError


def quantize_array_q4_0(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize array in contiguous 32-element blocks to q4_0 scales and packed nibbles."""
    raise NotImplementedError


def dequantize_array_q4_0(scales: np.ndarray, packed: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Dequantize scales and packed uint8 nibbles back into original array shape."""
    raise NotImplementedError
