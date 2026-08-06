import math

def kmeans_pp_seed(X: list[list[float]], n_clusters: int, rng_stream: list[float]) -> list[int]:
    """
    Correct implementation of K-Means++ seeding that consumes rng_stream.
    """
    n_samples = len(X)
    n_features = len(X[0])
    indices = [0] * n_clusters

    first_idx = int(math.floor(rng_stream[0] * n_samples))
    indices[0] = first_idx

    dists = [float('inf')] * n_samples
    first_center = X[first_idx]
    for i in range(n_samples):
        sq_dist = 0.0
        for j in range(n_features):
            d = float(X[i][j]) - float(first_center[j])
            sq_dist += d * d
        if sq_dist < dists[i]:
            dists[i] = sq_dist

    for t in range(1, n_clusters):
        total = 0.0
        for i in range(n_samples):
            total += dists[i]

        if total == 0.0:
            idx = int(math.floor(rng_stream[t] * n_samples))
        else:
            val = float(rng_stream[t])
            cum = 0.0
            idx = n_samples - 1
            for i in range(n_samples):
                cum += dists[i] / total
                if cum >= val:
                    idx = i
                    break
        indices[t] = idx

        new_center = X[idx]
        for i in range(n_samples):
            sq_dist = 0.0
            for j in range(n_features):
                d = float(X[i][j]) - float(new_center[j])
                sq_dist += d * d
            if sq_dist < dists[i]:
                dists[i] = sq_dist

    return indices
