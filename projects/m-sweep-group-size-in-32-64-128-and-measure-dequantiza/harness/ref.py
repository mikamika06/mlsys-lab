"""Reference oracle logic for test verification."""

import numpy as np


def generate_test_weights(shape=(256, 128), seed=1337):
    """Generate synthetic weight matrix for benchmark evaluation."""
    rng = np.random.RandomState(seed)
    return rng.randn(*shape).astype(np.float32)


def quantize_affine(weights: np.ndarray, group_size: int, bits: int = 4):
    """Reference affine quantizer."""
    orig_shape = weights.shape
    flat = weights.astype(np.float32).reshape(-1)
    n = flat.size
    pad_len = (group_size - (n % group_size)) % group_size
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), mode="constant", constant_values=0.0)

    num_groups = flat.size // group_size
    grouped = flat.reshape(num_groups, group_size)

    min_val = grouped.min(axis=1, keepdims=True)
    max_val = grouped.max(axis=1, keepdims=True)

    qmax = (1 << bits) - 1
    scales = np.where(max_val > min_val, (max_val - min_val) / float(qmax), 1.0).astype(np.float32)
    biases = min_val.astype(np.float32)

    qweight = np.clip(np.round((grouped - biases) / scales), 0, qmax).astype(np.uint8)

    if pad_len > 0:
        flat_q = qweight.reshape(-1)[:n]
        qweight = flat_q.reshape(-1)
    else:
        qweight = qweight.reshape(-1)

    return qweight.reshape(orig_shape), scales.reshape(-1), biases.reshape(-1)


def dequantize_affine(
    qweight: np.ndarray, scales: np.ndarray, biases: np.ndarray, group_size: int
) -> np.ndarray:
    """Reference affine dequantizer."""
    orig_shape = qweight.shape
    flat_q = qweight.astype(np.float32).reshape(-1)
    n = flat_q.size

    pad_len = (group_size - (n % group_size)) % group_size
    if pad_len > 0:
        flat_q = np.pad(flat_q, (0, pad_len), mode="constant", constant_values=0.0)

    num_groups = flat_q.size // group_size
    grouped_q = flat_q.reshape(num_groups, group_size)

    sc = scales.reshape(num_groups, 1)
    bi = biases.reshape(num_groups, 1)

    dequant = grouped_q * sc + bi
    flat_dequant = dequant.reshape(-1)[:n]
    return flat_dequant.reshape(orig_shape)


def pack_uint4_pair(uint4_array: np.ndarray) -> np.ndarray:
    """Reference 4-bit packing."""
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
    """Reference 4-bit unpacking and dequantization."""
    flat_packed = packed_weights.reshape(-1)
    low = flat_packed & 0x0F
    high = (flat_packed >> 4) & 0x0F

    unpacked = np.empty(flat_packed.size * 2, dtype=np.uint8)
    unpacked[0::2] = low
    unpacked[1::2] = high

    total_elements = int(np.prod(original_shape))
    unpacked = unpacked[:total_elements].reshape(original_shape)

    return dequantize_affine(unpacked, scales, biases, group_size=group_size)


def sweep_group_size_mse(weights: np.ndarray, group_sizes=(32, 64, 128), bits: int = 4) -> dict:
    """Reference group size sweep."""
    res = {}
    for gs in group_sizes:
        qw, sc, bi = quantize_affine(weights, group_size=gs, bits=bits)
        deq = dequantize_affine(qw, sc, bi, group_size=gs)
        mse = float(np.mean((weights.astype(np.float32) - deq) ** 2))
        res[gs] = mse
    return res


def compare_bit_widths(weights: np.ndarray, group_size: int = 64) -> dict:
    """Reference comparison for 4-bit vs 8-bit."""
    w_fp32 = weights.astype(np.float32)
    n_elems = w_fp32.size

    q4, s4, b4 = quantize_affine(w_fp32, group_size=group_size, bits=4)
    deq4 = dequantize_affine(q4, s4, b4, group_size=group_size)
    mse4 = float(np.mean((w_fp32 - deq4) ** 2))
    packed_q4 = pack_uint4_pair(q4)
    bytes4 = packed_q4.nbytes + s4.nbytes + b4.nbytes

    q8, s8, b8 = quantize_affine(w_fp32, group_size=group_size, bits=8)
    deq8 = dequantize_affine(q8, s8, b8, group_size=group_size)
    mse8 = float(np.mean((w_fp32 - deq8) ** 2))
    bytes8 = q8.nbytes + s8.nbytes + b8.nbytes

    fp32_bytes = w_fp32.nbytes

    return {
        "4bit": {
            "bytes": int(bytes4),
            "mse": mse4,
            "compression_ratio": float(fp32_bytes / bytes4),
        },
        "8bit": {
            "bytes": int(bytes8),
            "mse": mse8,
            "compression_ratio": float(fp32_bytes / bytes8),
        },
    }
