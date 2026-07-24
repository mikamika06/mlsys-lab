import numpy as np

def inertia_sequence(X: np.ndarray, n_clusters: int, max_iter: int = 10) -> list[float]:
    rng = np.random.default_rng(0)
    centroids = X[rng.choice(len(X), size=n_clusters, replace=False)]
    seq = []
    for _ in range(max_iter):
        # assignment
        diff = X[:, None, :] - centroids[None, :, :]
        d2 = np.sum(diff**2, axis=2)
        labels = np.argmin(d2, axis=1)

        # inertia
        inertia = np.sum((X - centroids[labels])**2)
        seq.append(float(inertia))

        # update centroids
        counts = np.bincount(labels, minlength=n_clusters)
        sums = np.array([np.bincount(labels, weights=X[:, d], minlength=n_clusters) for d in range(X.shape[1])]).T
        new_centroids = sums / counts[:, None]
        mask = counts == 0
        new_centroids[mask] = centroids[mask]

        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return seq
