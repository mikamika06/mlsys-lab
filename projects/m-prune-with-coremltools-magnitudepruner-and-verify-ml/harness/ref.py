import numpy as np


def get_test_models():
    np.random.seed(42)
    return [
        np.random.randn(32, 32),
        np.random.randn(64, 64),
        np.random.randn(16, 128)
    ]


def oracle_prune(w, sparsity=0.5):
    flat = np.abs(w.flatten())
    if sparsity <= 0.0:
        return w.copy(), w.nbytes
    if sparsity >= 1.0:
        return np.zeros_like(w), 0
    k = int(np.floor((1.0 - sparsity) * flat.size))
    if k <= 0:
        return np.zeros_like(w), 0
    thresh = np.partition(flat, flat.size - k)[flat.size - k]
    mask = np.abs(w) >= thresh
    pruned = w * mask
    non_zeros = np.count_nonzero(pruned)
    bytes_size = int(non_zeros * 4 + w.size * 0.2)
    return pruned, bytes_size


def oracle_chain(w, sparsity=0.5, n_bits=4):
    pruned, _ = oracle_prune(w, sparsity)
    flat = pruned.flatten()
    n_clusters = 2 ** n_bits
    vmin, vmax = np.min(flat), np.max(flat)
    if vmin == vmax:
        quantized = pruned.copy()
    else:
        edges = np.linspace(vmin, vmax, n_clusters + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        idxs = np.digitize(flat, edges[1:-1])
        quantized = centers[idxs].reshape(w.shape)
    nz = np.count_nonzero(quantized)
    combined_bytes = int(np.ceil(nz * n_bits / 8.0) + (2 ** n_bits) * 4 + w.size * 0.1)
    return quantized, combined_bytes


def oracle_derive(shape, sparsity=0.5, n_bits=4):
    total_elements = int(np.prod(shape))
    active_elements = int(total_elements * (1.0 - sparsity))
    data_bytes = int(np.ceil(active_elements * n_bits / 8.0))
    palette_bytes = (2 ** n_bits) * 4
    overhead = int(total_elements * 0.05)
    return data_bytes + palette_bytes + overhead
