import numpy as np


def generate_nm_mask(tensor, n=2, m=4):
    arr = np.array(tensor, dtype=float)
    shape = arr.shape
    reshaped = arr.reshape(-1, m)
    abs_vals = np.abs(reshaped)
    mask = np.zeros_like(reshaped)
    top_indices = np.argsort(abs_vals, axis=1)[:, -n:]
    rows = np.arange(reshaped.shape[0])[:, None]
    mask[rows, top_indices] = 1.0
    return mask.reshape(shape)
