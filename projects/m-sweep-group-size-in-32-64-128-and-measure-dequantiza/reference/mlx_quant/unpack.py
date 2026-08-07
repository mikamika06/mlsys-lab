"""Packed 4-bit weight unpacker and hand dequantizer."""

import numpy as np
from mlx_quant.sweep import dequantize_affine


def pack_uint4_pair(uint4_array: np.ndarray) -> np.ndarray:
    """Pack array of uint4 values (0-15) into uint8 bytes (2 values per byte)."""
    flat = uint4_array.astype(np.uint8).reshape(-1)
    if flat.size % 2 != 0:
        flat = np.pad(flat, (0, 1), mode="constant", constant_values=0)
    low = flat[0::2] & 0x0F
    high = (flat[1::2] & 0x0F) << 4
    return (low | high).astype(np.uint8)


def unpack_and_dequantize_4bit(
    packed_weights: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    group_size: int,
    original_shape: tuple,
) -> np.ndarray:
    """Unpack 4-bit packed uint8 array and dequantize to float array."""
    flat_packed = packed_weights.reshape(-1)
    low = flat_packed & 0x0F
    high = (flat_packed >> 4) & 0x0F

    unpacked = np.empty(flat_packed.size * 2, dtype=np.uint8)
    unpacked[0::2] = low
    unpacked[1::2] = high

    total_elements = int(np.prod(original_shape))
    unpacked = unpacked[:total_elements].reshape(original_shape)

    return dequantize_affine(unpacked, scales, biases, group_size=group_size)
