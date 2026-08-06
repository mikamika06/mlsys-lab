import numpy as np

def magnitude_mask(W, sparsity):
    k = int(W.shape[1] * sparsity)
    mask = np.ones_like(W, dtype=bool)
    if k == 0:
        return mask
    for i in range(W.shape[0]):
        idx = np.argsort(np.abs(W[i]))[:k]
        mask[i, idx] = False
    return mask

def wanda_mask(W, X, sparsity):
    k = int(W.shape[1] * sparsity)
    mask = np.ones_like(W, dtype=bool)
    if k == 0:
        return mask
    x_norm = np.linalg.norm(X, axis=0)
    score = np.abs(W) * x_norm
    for i in range(W.shape[0]):
        idx = np.argsort(score[i])[:k]
        mask[i, idx] = False
    return mask

def mask_recall(mask_a, mask_b):
    kept_a = np.sum(mask_a)
    if kept_a == 0:
        return 1.0
    return float(np.sum(mask_a & mask_b)) / kept_a
