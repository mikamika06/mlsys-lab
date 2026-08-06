import math
import numpy as np

def l1_unstructured_mask(weight: np.ndarray, amount: float) -> np.ndarray:
    flat = weight.ravel()
    abs_flat = np.empty_like(flat)
    for i in range(flat.size):
        val = flat[i]
        if val < 0:
            abs_flat[i] = -val
        else:
            abs_flat[i] = val
    k = int(math.floor(amount * flat.size))
    if k <= 0:
        return np.ones_like(weight, dtype=bool)
    indices = []
    for i in range(flat.size):
        indices.append(i)
    sorted_idx = np.array(sorted(indices, key=lambda i: abs_flat[i]), dtype=np.intp)
    mask_flat = np.ones(flat.size, dtype=bool)
    for i in range(k):
        mask_flat[sorted_idx[i]] = False
    return mask_flat.reshape(weight.shape)
