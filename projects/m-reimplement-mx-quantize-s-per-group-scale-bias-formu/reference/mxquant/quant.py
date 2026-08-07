import numpy as np


def quantize_per_group(weights, group_size, bits):
    if weights.shape[1] % group_size != 0:
        raise ValueError("Weight shape dimension must be divisible by group_size")

    in_features = weights.shape[1]
    num_groups = in_features // group_size
    reshaped = weights.reshape(weights.shape[0], num_groups, group_size)

    max_val = (1 << (bits - 1)) - 1
    min_val = -(1 << (bits - 1))

    w_min = reshaped.min(axis=-1, keepdims=True)
    w_max = reshaped.max(axis=-1, keepdims=True)

    scale = (w_max - w_min) / (max_val - min_val)
    scale = np.where(scale == 0, 1e-5, scale)
    bias = w_min

    quantized = np.clip(np.round((reshaped - bias) / scale), min_val, max_val)
    quantized = quantized.astype(np.int32)

    return quantized, scale, bias


def quantize_linear(weight_matrix, group_size, bits):
    q, s, b = quantize_per_group(weight_matrix, group_size, bits)
    original_bytes = weight_matrix.nbytes
    packed_bits = q.size * bits
    packed_bytes = packed_bits // 8 + (1 if packed_bits % 8 != 0 else 0)
    scale_bytes = s.nbytes
    bias_bytes = b.nbytes
    total_new_bytes = packed_bytes + scale_bytes + bias_bytes
    ratio = original_bytes / total_new_bytes
    return q, s, b, ratio
