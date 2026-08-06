import math
import numpy as np


def predict_kmeans_convergence(
    X: np.ndarray, k: int, max_iter: int = 300, tol: float = 1e-4
) -> tuple[int, np.ndarray]:
    n_samples, n_features = X.shape
    rng = np.random.default_rng(0)
    init_indices = rng.choice(n_samples, size=k, replace=False)

    centroids = np.empty((k, n_features), dtype=X.dtype)
    for j in range(k):
        idx = init_indices[j]
        for f in range(n_features):
            centroids[j, f] = X[idx, f]

    labels = np.empty(n_samples, dtype=int)

    for it in range(max_iter):
        new_labels = np.empty(n_samples, dtype=int)
        for i in range(n_samples):
            min_dist = float("inf")
            best_j = 0
            for j in range(k):
                sq_dist = 0.0
                for f in range(n_features):
                    diff = float(X[i, f]) - float(centroids[j, f])
                    sq_dist += diff * diff
                dist = math.sqrt(sq_dist)
                if dist < min_dist:
                    min_dist = dist
                    best_j = j
            new_labels[i] = best_j

        counts = [0] * k
        for i in range(n_samples):
            counts[new_labels[i]] += 1

        new_centroids = np.empty((k, n_features), dtype=X.dtype)
        for j in range(k):
            if counts[j] > 0:
                for f in range(n_features):
                    feat_sum = 0.0
                    for i in range(n_samples):
                        if new_labels[i] == j:
                            feat_sum += float(X[i, f])
                    new_centroids[j, f] = feat_sum / counts[j]
            else:
                for f in range(n_features):
                    new_centroids[j, f] = centroids[j, f]

        max_shift = 0.0
        for j in range(k):
            sq_diff_sum = 0.0
            for f in range(n_features):
                diff = float(new_centroids[j, f]) - float(centroids[j, f])
                sq_diff_sum += diff * diff
            shift_j = math.sqrt(sq_diff_sum)
            if shift_j > max_shift:
                max_shift = shift_j

        centroids = new_centroids
        labels = new_labels

        if max_shift < tol:
            return it + 1, labels.copy()

    return max_iter, labels.copy()
