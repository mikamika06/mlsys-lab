import numpy as np


def quantize_q4_0(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantizes an array along its last dimension in blocks of 32 using q4_0."""
    raise NotImplementedError


def dequantize_q4_0(scales: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Dequantizes q4_0 scales and packed nibbles back to float32 array."""
    raise NotImplementedError


def quantized_k_dot(q: np.ndarray, scales: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Computes dot product between query q and quantized K tensor."""
    raise NotImplementedError


def k_cache_bytes(shape: tuple[int, ...], dtype_str: str) -> int:
    """Returns total bytes required for a K cache tensor of given shape and format."""
    raise NotImplementedError
