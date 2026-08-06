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
    n_samples, n_features = X.shape
    centroids = np.zeros((k, n_features), dtype=np.float64)
    for i in range(k):
        for j in range(n_features):
            centroids[i, j] = float(X[i, j])

    for it in range(1, n_iter + 1):
        idx = rng.choice(n_samples, size=batch_size, replace=True)

        batch = np.zeros((batch_size, n_features), dtype=X.dtype)
        for b in range(batch_size):
            for j in range(n_features):
                batch[b, j] = X[idx[b], j]

        labels = [0] * batch_size
        for b in range(batch_size):
            min_dist = float('inf')
            best_c = 0
            for c in range(k):
                d_sq = 0.0
                for j in range(n_features):
                    diff = float(batch[b, j]) - centroids[c, j]
                    d_sq += diff * diff
                if d_sq < min_dist:
                    min_dist = d_sq
                    best_c = c
            labels[b] = best_c

        new_centroids = np.zeros((k, n_features), dtype=np.float64)
        for c in range(k):
            count = 0
            for b in range(batch_size):
                if labels[b] == c:
                    count += 1

            if count > 0:
                for j in range(n_features):
                    sum_val = 0.0
                    for b in range(batch_size):
                        if labels[b] == c:
                            sum_val += float(batch[b, j])
                    new_centroids[c, j] = sum_val / count
            else:
                for j in range(n_features):
                    new_centroids[c, j] = centroids[c, j]

        for c in range(k):
            for j in range(n_features):
                centroids[c, j] = (centroids[c, j] * (it - 1) + new_centroids[c, j]) / it

    return centroids
