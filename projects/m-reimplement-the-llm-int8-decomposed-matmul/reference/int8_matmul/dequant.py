import numpy as np


def derive_vector_scales(tensor):
    max_vals = np.max(np.abs(tensor), axis=-1, keepdims=True)
    scales = max_vals / 127.0
    scales = np.where(scales == 0, 1e-8, scales)
    return scales
