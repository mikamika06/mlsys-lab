"""Packed 4-bit weight unpacker and hand dequantizer."""

import numpy as np


def pack_uint4_pair(uint4_array: np.ndarray) -> np.ndarray:
    """Pack array of uint4 values (0-15) into uint8 bytes (2 values per byte)."""
    raise NotImplementedError


def unpack_and_dequantize_4bit(
    packed_weights: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    group_size: int,
    original_shape: tuple,
) -> np.ndarray:
    """Unpack 4-bit packed uint8 array and dequantize to float array."""
    raise NotImplementedError
