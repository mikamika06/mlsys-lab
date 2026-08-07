import numpy as np


def pack_int4_groups(weights: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    flat = weights.flatten()
    n = len(flat)
    num_groups = (n + group_size - 1) // group_size
    padded_len = num_groups * group_size
    padded = np.zeros(padded_len, dtype=np.float32)
    padded[:n] = flat

    reshaped = padded.reshape(num_groups, group_size)
    max_vals = np.max(np.abs(reshaped), axis=1, keepdims=True)
    scales = np.where(max_vals == 0, 1e-5, max_vals / 7.0)

    quantized = np.clip(np.round(reshaped / scales), -8, 7).astype(np.int8)
    unsigned_nibbles = (quantized + 8).astype(np.uint8)

    packed_group_size = group_size // 2
    packed = np.zeros((num_groups, packed_group_size), dtype=np.uint8)
    for i in range(0, group_size, 2):
        low = unsigned_nibbles[:, i]
        high = unsigned_nibbles[:, i + 1]
        packed[:, i // 2] = (high << 4) | (low & 0x0F)

    return packed, scales.squeeze(1)


def unpack_int4_groups(packed: np.ndarray, scales: np.ndarray, group_size: int, original_shape: tuple[int, int]) -> np.ndarray:
    num_groups, packed_group_size = packed.shape
    unsigned_nibbles = np.zeros((num_groups, group_size), dtype=np.uint8)
    for i in range(packed_group_size):
        b = packed[:, i]
        unsigned_nibbles[:, 2 * i] = b & 0x0F
        unsigned_nibbles[:, 2 * i + 1] = (b >> 4) & 0x0F

    signed_vals = unsigned_nibbles.astype(np.int8) - 8
    dequantized = signed_vals.astype(np.float32) * scales[:, np.newaxis]
    flat = dequantized.flatten()
    total_elements = original_shape[0] * original_shape[1]
    return flat[:total_elements].reshape(original_shape)
