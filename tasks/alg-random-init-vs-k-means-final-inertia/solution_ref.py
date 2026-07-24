import numpy as np

def _kmeans(X, n_clusters, init, seed):
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    if init == "random":
        idx = rng.choice(n_samples, size=n_clusters, replace=False)
        centers = X[idx]
    elif init == "kpp":
        idx = [rng.integers(0, n_samples)]
        centers = X[idx]
        for _ in range(1, n_clusters):
            dists = np.min(np.sum((X[:, None] - centers[None])**2, axis=2), axis=1)
            probs = dists / dists.sum()
            idx.append(rng.choice(n_samples, p=probs))
            centers = X[idx]
    else:
        raise ValueError("init must be 'random' or 'kpp'")
    for _ in range(300):
        dist_sq = np.sum((X[:, None] - centers[None])**2, axis=2)
        labels = np.argmin(dist_sq, axis=1)
        new_centers = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                                for k in range(n_clusters)])
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers
    inertia = np.sum((X - centers[labels])**2)
    return float(inertia)

def compare_inertia(X: np.ndarray,
                    n_clusters: int,
                    seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    random_inertias = []
    kpp_inertias   = []
    for seed in seeds:
        random_inertias.append(_kmeans(X, n_clusters, "random", seed))
        kpp_inertias.append(_kmeans(X, n_clusters, "kpp", seed))
    return (np.array(random_inertias, dtype=np.float64),
            np.array(kpp_inertias,   dtype=np.float64))
