import numpy as np
from typing import Tuple

def lloyd_fixed_init(
    X: np.ndarray,
    init_centroids: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-4
) -> Tuple[np.ndarray, int]:
    centroids = init_centroids.astype(np.float64).copy()
    labels_prev = None
    for it in range(1, max_iter + 1):
        diff = X[:, None, :] - centroids[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        labels = np.argmin(dist_sq, axis=1)
        if labels_prev is not None and np.array_equal(labels, labels_prev):
            return labels.astype(np.int64), it - 1
        labels_prev = labels.copy()
        for i in range(centroids.shape[0]):
            mask = (labels == i)
            if np.any(mask):
                centroids[i] = X[mask].mean(axis=0)
    return labels.astype(np.int64), max_iter
