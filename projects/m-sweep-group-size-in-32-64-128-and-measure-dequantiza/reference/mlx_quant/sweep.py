"""Group size sweep and MSE calculation."""

import numpy as np


def quantize_affine(weights: np.ndarray, group_size: int, bits: int = 4):
    """Quantize weights with affine scale and bias per group."""
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

    qweight = qweight.reshape(orig_shape)
    scales = scales.reshape(-1)
    biases = biases.reshape(-1)

    return qweight, scales, biases


def dequantize_affine(
    qweight: np.ndarray, scales: np.ndarray, biases: np.ndarray, group_size: int
) -> np.ndarray:
    """Dequantize uint8 values back to float weights using scale and bias."""
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


def sweep_group_size_mse(weights: np.ndarray, group_sizes=(32, 64, 128), bits: int = 4) -> dict:
    """Sweep group sizes and compute dequantization MSE relative to weights."""
    res = {}
    for gs in group_sizes:
        qw, sc, bi = quantize_affine(weights, group_size=gs, bits=bits)
        deq = dequantize_affine(qw, sc, bi, group_size=gs)
        mse = float(np.mean((weights.astype(np.float32) - deq) ** 2))
        res[gs] = mse
    return res
