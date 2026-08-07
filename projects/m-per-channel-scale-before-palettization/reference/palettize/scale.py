import numpy as np


def per_channel_scale(weight: np.ndarray, nbits: int):
    if weight.ndim == 1:
        weight = weight[:, np.newaxis]
    min_val = np.min(weight, axis=1, keepdims=True)
    max_val = np.max(weight, axis=1, keepdims=True)
    levels = 2 ** nbits
    scale = (max_val - min_val) / (levels - 1)
    scale = np.where(scale == 0, 1.0, scale)
    zero_point = np.round(-min_val / scale)
    zero_point = np.clip(zero_point, 0, levels - 1)
    quantized = np.clip(np.round(weight / scale + zero_point), 0, levels - 1).astype(np.int32)
    return quantized, scale, zero_point
