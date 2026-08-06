import numpy as np


def compute_longrope_factors(head_dim, original_max_len, target_max_len, base=10000.0, short_factor=None, long_factor=None):
    scale = target_max_len / original_max_len
    dims = np.arange(0, head_dim, 2, dtype=np.float64)
    inv_freq = 1.0 / (base ** (dims / head_dim))

    if short_factor is not None and long_factor is not None:
        factors = np.where(scale <= 1.0, short_factor, long_factor)
    else:
        wavelengths = 2.0 * np.pi / inv_freq
        factors = np.ones(head_dim // 2, dtype=np.float64)
        for i, w in enumerate(wavelengths):
            if w > original_max_len:
                factors[i] = scale
            elif w < original_max_len / 4.0:
                factors[i] = 1.0
            else:
                ratio = (w - original_max_len / 4.0) / (original_max_len - original_max_len / 4.0)
                factors[i] = 1.0 + ratio * (scale - 1.0)

    return factors
