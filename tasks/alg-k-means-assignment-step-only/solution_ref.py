import numpy as np

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    diff = X[:, None, :] - centroids[None, :, :]
    dists = np.sum(diff**2, axis=2)
    labels = np.argmin(dists, axis=1)
    return labels
