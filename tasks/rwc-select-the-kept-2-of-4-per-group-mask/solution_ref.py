import numpy as np


def select_top2_mask(weights: np.ndarray) -> np.ndarray:
    """Select top 2 elements by absolute value in each group of 4."""
    mask = np.zeros_like(weights, dtype=bool)
    flat_weights = weights.reshape(-1)
    flat_mask = mask.reshape(-1)
    n = len(flat_weights)

    for i in range(0, n, 4):
        vals = [0.0, 0.0, 0.0, 0.0]
        for j in range(4):
            w = flat_weights[i + j]
            vals[j] = -w if w < 0 else w

        idxs = [0, 1, 2, 3]
        for j in range(1, 4):
            key_idx = idxs[j]
            key_val = vals[key_idx]
            k = j - 1
            while k >= 0 and vals[idxs[k]] < key_val:
                idxs[k + 1] = idxs[k]
                k -= 1
            idxs[k + 1] = key_idx

        flat_mask[i + idxs[0]] = True
        flat_mask[i + idxs[1]] = True

    return mask
