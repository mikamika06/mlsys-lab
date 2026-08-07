import numpy as np


def compute_per_row_symmetric_scales(weight: np.ndarray) -> np.ndarray:
    max_vals = np.max(np.abs(weight), axis=1, keepdims=True)
    scales = max_vals / 127.0
    scales = np.where(scales == 0, 1e-8, scales)
    return scales
