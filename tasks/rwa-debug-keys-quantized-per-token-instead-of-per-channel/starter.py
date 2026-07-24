import numpy as np


def quantize_keys_per_channel(K, bits=4):
    # TODO: this incorrectly computes scales along the token axis.
    # Large channels dominate each token scale and reduce precision in
    # smaller key channels.
    K = np.asarray(K, dtype=np.float64)
    levels = 2 ** (bits - 1) - 1
    scale = np.max(np.abs(K), axis=1, keepdims=True) / levels
    scale = np.where(scale == 0, 1.0, scale)
    q = np.round(K / scale)
    return q * scale
