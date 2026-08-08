import numpy as np


def k_means_quantize(weights: np.ndarray, n_bits: int, max_iter: int = 20, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    flat = weights.astype(np.float32).ravel()
    k = 1 << n_bits
    if flat.size <= k:
        codebook = np.sort(flat)
        indices = np.searchsorted(codebook, flat).astype(np.uint32)
        return codebook, indices.reshape(weights.shape)

    min_val, max_val = float(flat.min()), float(flat.max())
    centroids = np.linspace(min_val, max_val, k, dtype=np.float32)

    for _ in range(max_iter):
        dists = np.abs(flat[:, None] - centroids[None, :])
        labels = np.argmin(dists, axis=1)
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_centroids[i] = flat[mask].mean()
            else:
                new_centroids[i] = centroids[i]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    dists = np.abs(flat[:, None] - centroids[None, :])
    labels = np.argmin(dists, axis=1).astype(np.uint32)
    return centroids.astype(np.float32), labels.reshape(weights.shape)
