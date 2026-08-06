import numpy as np


def fp32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """Converts a float32 array into uint16 bit patterns representing bf16 with RNE rounding."""
    raise NotImplementedError


def bf16_bits_to_fp32(bits: np.ndarray) -> np.ndarray:
    """Reconstructs a float32 array from uint16 bf16 bit patterns."""
    raise NotImplementedError


def round_fp32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Rounds float32 values to bf16 precision and returns them as float32."""
    raise NotImplementedError
