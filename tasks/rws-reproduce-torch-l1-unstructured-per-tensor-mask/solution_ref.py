import numpy as np

def l1_unstructured_mask(weight: np.ndarray, amount: float) -> np.ndarray:
    flat = weight.ravel()
    abs_flat = np.abs(flat)
    k = int(np.floor(amount * flat.size))
    if k <= 0:
        return np.ones_like(weight, dtype=bool)
    sorted_idx = np.argsort(abs_flat)
    mask_flat = np.ones_like(flat, dtype=bool)
    mask_flat[sorted_idx[:k]] = False
    return mask_flat.reshape(weight.shape)
