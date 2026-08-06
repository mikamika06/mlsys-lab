import numpy as np


def kmeans_palettize(tensor: np.ndarray, n_bits: int, vector_length: int = 1) -> tuple[np.ndarray, np.ndarray]:
    k = 2 ** n_bits
    flat = tensor.astype(np.float32).ravel()
    if vector_length > 1:
        pad_len = (vector_length - (flat.size % vector_length)) % vector_length
        if pad_len > 0:
            flat = np.pad(flat, (0, pad_len), mode="constant")
        blocks = flat.reshape(-1, vector_length)
    else:
        blocks = flat[:, np.newaxis]

    np.random.seed(42)
    idx = np.random.choice(blocks.shape[0], size=min(k, blocks.shape[0]), replace=False)
    centroids = blocks[idx].copy()
    if centroids.shape[0] < k:
        padding = np.zeros((k - centroids.shape[0], vector_length), dtype=np.float32)
        centroids = np.vstack([centroids, padding])

    for _ in range(10):
        dists = np.sum((blocks[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centroids = np.array([blocks[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)], dtype=np.float32)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    dists = np.sum((blocks[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1).astype(np.uint32)
    return centroids, labels
