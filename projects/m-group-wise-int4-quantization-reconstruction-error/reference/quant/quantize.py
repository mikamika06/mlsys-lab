import numpy as np


def quantize_dequantize_int4(tensor: np.ndarray, group_size: int = 32):
    shape = tensor.shape
    flat = tensor.flatten()
    padded_len = ((len(flat) + group_size - 1) // group_size) * group_size
    padded = np.zeros(padded_len, dtype=np.float32)
    padded[:len(flat)] = flat
    groups = padded.reshape(-1, group_size)
    dequant_groups = np.zeros_like(groups)
    for i, g in enumerate(groups):
        g_min = np.min(g)
        g_max = np.max(g)
        limit = max(abs(g_min), abs(g_max), 1e-8)
        scale = limit / 7.0
        q = np.clip(np.round(g / scale), -8, 7)
        dequant_groups[i] = q * scale
    dequant_flat = dequant_groups.flatten()[:len(flat)]
    return dequant_flat.reshape(shape)
