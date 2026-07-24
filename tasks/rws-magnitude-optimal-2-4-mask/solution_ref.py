import numpy as np

def magnitude_optimal_2to4_mask(weights: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask that keeps the two largest‑magnitude weights in each
    consecutive block of four columns.
    """
    if weights.shape[-1] % 4 != 0:
        raise ValueError("last dimension must be a multiple of 4")
    abs_w = np.abs(weights)
    reshaped = abs_w.reshape(-1, 4)
    # indices of two largest per row
    idx = np.argpartition(reshaped, -2, axis=1)[:, -2:]
    mask_flat = np.zeros_like(reshaped, dtype=bool)
    rows = np.arange(mask_flat.shape[0])[:, None]
    mask_flat[rows, idx] = True
    return mask_flat.reshape(weights.shape)
