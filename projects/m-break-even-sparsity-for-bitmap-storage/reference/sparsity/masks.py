import numpy as np


def compute_mask(weights, sparsity, mode="global"):
    flat = np.abs(weights)
    if mode == "global":
        k = int(np.floor(sparsity * flat.size))
        if k <= 0:
            return np.ones_like(weights, dtype=bool)
        if k >= flat.size:
            return np.zeros_like(weights, dtype=bool)
        thresh = np.partition(flat.ravel(), k)[k]
        return flat > thresh
    elif mode == "per-layer":
        mask = np.empty_like(weights, dtype=bool)
        if weights.ndim == 1:
            k = int(np.floor(sparsity * weights.size))
            thresh = np.partition(np.abs(weights), k)[k]
            return np.abs(weights) > thresh
        for i in range(weights.shape[0]):
            sub = weights[i]
            k = int(np.floor(sparsity * sub.size))
            if k <= 0:
                mask[i] = True
            elif k >= sub.size:
                mask[i] = False
            else:
                thresh = np.partition(np.abs(sub).ravel(), k)[k]
                mask[i] = np.abs(sub) > thresh
        return mask
    else:
        raise ValueError(f"Unknown mode {mode}")
