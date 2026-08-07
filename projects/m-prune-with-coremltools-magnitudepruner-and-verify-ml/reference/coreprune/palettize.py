import numpy as np
from coreprune.prune import prune_weights


def palettize_weights(weights, n_bits=4):
    flat = weights.flatten()
    n_clusters = 2 ** n_bits
    vmin, vmax = np.min(flat), np.max(flat)
    if vmin == vmax:
        return weights.copy(), int(weights.size * n_bits / 8)
    edges = np.linspace(vmin, vmax, n_clusters + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    idxs = np.digitize(flat, edges[1:-1])
    quantized = centers[idxs].reshape(weights.shape)
    bytes_size = int(np.ceil(weights.size * n_bits / 8.0) + n_clusters * 4)
    return quantized, bytes_size


def chain_prune_palettize(weights, sparsity=0.5, n_bits=4):
    pruned, _ = prune_weights(weights, sparsity)
    quantized, _ = palettize_weights(pruned, n_bits)
    nz = np.count_nonzero(quantized)
    combined_bytes = int(np.ceil(nz * n_bits / 8.0) + (2 ** n_bits) * 4 + weights.size * 0.1)
    return quantized, combined_bytes
