import numpy as np

def run_indexing_kernel(size):
    data = np.arange(size, dtype=np.float32)
    out = data + 1.0
    return out

def run_sum_reduction_kernel(arr, math_mode="safe"):
    return float(np.sum(arr))

def run_dequant_kernel(packed, scales, biases):
    unpacked_low = packed & 0x0F
    unpacked_high = (packed >> 4) & 0x0F
    unpacked = np.stack([unpacked_low, unpacked_high], axis=-1).reshape(*packed.shape[:-1], -1)
    return unpacked.astype(np.float32) * scales[..., None] + biases[..., None]
