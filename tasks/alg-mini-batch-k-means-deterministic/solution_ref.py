import numpy as np

def mini_batch_kmeans(
    X: np.ndarray,
    k: int,
    batch_size: int,
    n_iter: int,
    seed: int = 0
) -> np.ndarray:
    """
    Deterministic mini‑batch k‑means.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Data points.
    k : int
        Number of clusters.
    batch_size : int
        Size of the random batch at each iteration.
    n_iter : int
        Number of iterations to run.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    centroids : ndarray, shape (k, n_features)
        Final cluster centroids after `n_iter` updates.
    """
    rng = np.random.default_rng(seed)
    centroids = X[:k].astype(np.float64).copy()

    for it in range(1, n_iter + 1):
        # Sample a batch with replacement
        idx = rng.choice(len(X), size=batch_size, replace=True)
        batch = X[idx]

        # Compute squared Euclidean distances to centroids
        dists = np.sum((batch[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        # Batch means per centroid (keep old value if no assignment)
        new_centroids = centroids.copy()
        for i in range(k):
            mask = (labels == i)
            if np.any(mask):
                new_centroids[i] = np.mean(batch[mask], axis=0)

        # Incremental moving average
        centroids = (centroids * (it - 1) + new_centroids) / it

    return centroids
