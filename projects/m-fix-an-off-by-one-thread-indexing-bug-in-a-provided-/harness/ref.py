import numpy as np

def compute_reference_indexing(size):
    arr = np.arange(size, dtype=np.float32)
    return arr + 1.0

def compute_reference_sum(arr):
    return float(np.sum(arr))

def compute_reference_dequant(packed, scales, biases):
    unpacked_low = packed & 0x0F
    unpacked_high = (packed >> 4) & 0x0F
    unpacked = np.stack([unpacked_low, unpacked_high], axis=-1).reshape(*packed.shape[:-1], -1)
    return unpacked.astype(np.float32) * scales[..., None] + biases[..., None]
