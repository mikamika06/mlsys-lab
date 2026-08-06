import numpy as np

def global_magnitude_prune(weights, sparsity):
    flat = np.concatenate([np.abs(w).ravel() for w in weights.values()])
    k = int(round((1.0 - sparsity) * flat.size))
    if k <= 0:
        thresh = np.inf
    elif k >= flat.size:
        thresh = -1.0
    else:
        partitioned = np.partition(flat, flat.size - k)
        thresh = partitioned[flat.size - k]
    masks = {}
    for name, w in weights.items():
        masks[name] = (np.abs(w) >= thresh).astype(np.float32)
    return masks
