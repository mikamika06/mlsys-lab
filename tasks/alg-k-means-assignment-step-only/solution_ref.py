import numpy as np

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    n, d = X.shape
    k = centroids.shape[0]
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        best_j = 0
        best_dist = float("inf")
        for j in range(k):
            dist = 0.0
            for m in range(d):
                diff = X[i, m] - centroids[j, m]
                dist += diff * diff
            if dist < best_dist:
                best_dist = dist
                best_j = j
        labels[i] = best_j
    return labels
