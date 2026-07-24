import numpy as np

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    X_norm = np.sum(X**2, axis=1)[:, None]
    C_norm = np.sum(centroids**2, axis=1)[None, :]
    cross = X @ centroids.T
    dists_sq = X_norm + C_norm - 2*cross
    return np.argmin(dists_sq, axis=1)
