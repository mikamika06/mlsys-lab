import numpy as np

def predict_kmeans_convergence(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    tol: float = 1e-4
) -> tuple[int, np.ndarray]:
    rng = np.random.default_rng(0)
    centroids = X[rng.choice(len(X), size=k, replace=False)]
    labels = np.empty(len(X), dtype=int)
    for it in range(max_iter):
        # Assignment step: compute squared Euclidean distances
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)
        new_labels = np.argmin(dists, axis=1)
        # Update step: recompute centroids as means of assigned points
        new_centroids = np.array([
            X[new_labels == j].mean(axis=0) if np.any(new_labels == j) else centroids[j]
            for j in range(k)
        ])
        shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
        centroids = new_centroids
        labels = new_labels
        if shift < tol:
            return it + 1, labels.copy()
    return max_iter, labels.copy()
