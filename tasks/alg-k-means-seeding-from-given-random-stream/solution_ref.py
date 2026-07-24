import numpy as np

def kmeans_pp_seed(X: np.ndarray, n_clusters: int, rng_stream: np.ndarray) -> np.ndarray:
    """
    Correct implementation of K-Means++ seeding that consumes rng_stream.
    """
    n_samples = X.shape[0]
    indices = np.empty(n_clusters, dtype=np.int64)

    # First center uniformly at random
    first_idx = int(np.floor(rng_stream[0] * n_samples))
    indices[0] = first_idx

    # Distances to nearest chosen center for each point
    dists = np.full(n_samples, np.inf, dtype=np.float64)
    diff = X - X[first_idx]
    dists = np.minimum(dists, np.sum(diff * diff, axis=1))

    for t in range(1, n_clusters):
        total = dists.sum()
        if total == 0.0:
            idx = int(np.floor(rng_stream[t] * n_samples))
        else:
            cum = np.cumsum(dists) / total
            val = rng_stream[t]
            idx = np.searchsorted(cum, val)
        indices[t] = idx

        diff = X - X[idx]
        dists = np.minimum(dists, np.sum(diff * diff, axis=1))

    return indices
