import numpy as np


def prune_weights(weights, sparsity=0.5):
    flat = np.abs(weights.flatten())
    if sparsity <= 0.0:
        return weights.copy(), weights.nbytes
    if sparsity >= 1.0:
        return np.zeros_like(weights), 0
    k = int(np.floor((1.0 - sparsity) * flat.size))
    if k <= 0:
        return np.zeros_like(weights), 0
    thresh = np.partition(flat, flat.size - k)[flat.size - k]
    mask = np.abs(weights) >= thresh
    pruned = weights * mask
    non_zeros = np.count_nonzero(pruned)
    bytes_size = int(non_zeros * 4 + weights.size * 0.2)
    return pruned, bytes_size
