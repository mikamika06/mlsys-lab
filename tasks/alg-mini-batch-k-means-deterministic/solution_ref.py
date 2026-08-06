import random

def mini_batch_kmeans(
    X: list[list[float]],
    k: int,
    batch_size: int,
    n_iter: int,
    seed: int = 0
) -> list[list[float]]:
    """
    Deterministic mini‑batch k‑means.

    Parameters
    ----------
    X : list of list of float
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
    centroids : list of list of float
        Final cluster centroids after `n_iter` updates.
    """
    rng = random.Random(seed)
    n_samples = len(X)
    n_features = len(X[0])

    centroids = [[0.0] * n_features for _ in range(k)]
    for i in range(k):
        for j in range(n_features):
            centroids[i][j] = float(X[i][j])

    for it in range(1, n_iter + 1):
        idx = [rng.randrange(n_samples) for _ in range(batch_size)]

        batch = [[0.0] * n_features for _ in range(batch_size)]
        for b in range(batch_size):
            for j in range(n_features):
                batch[b][j] = float(X[idx[b]][j])

        labels = [0] * batch_size
        for b in range(batch_size):
            min_dist = float('inf')
            best_c = 0
            for c in range(k):
                d_sq = 0.0
                for j in range(n_features):
                    diff = batch[b][j] - centroids[c][j]
                    d_sq += diff * diff
                if d_sq < min_dist:
                    min_dist = d_sq
                    best_c = c
            labels[b] = best_c

        new_centroids = [[0.0] * n_features for _ in range(k)]
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
                            sum_val += batch[b][j]
                    new_centroids[c][j] = sum_val / count
            else:
                for j in range(n_features):
                    new_centroids[c][j] = centroids[c][j]

        for c in range(k):
            for j in range(n_features):
                centroids[c][j] = (centroids[c][j] * (it - 1) + new_centroids[c][j]) / it

    return centroids
