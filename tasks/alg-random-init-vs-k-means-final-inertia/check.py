import numpy as np

def _kmeans(X, n_clusters, init, seed):
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    # initialise centres
    if init == "random":
        idx = rng.choice(n_samples, size=n_clusters, replace=False)
        centers = X[idx]
    elif init == "kpp":
        idx = [rng.integers(0, n_samples)]
        centers = X[idx]
        for _ in range(1, n_clusters):
            # distances to nearest centre
            dists = np.min(np.sum((X[:, None] - centers[None])**2, axis=2), axis=1)
            probs = dists / dists.sum()
            idx.append(rng.choice(n_samples, p=probs))
            centers = X[idx]
    else:
        raise ValueError("init must be 'random' or 'kpp'")
    # Lloyd's algorithm
    for _ in range(300):
        # assignment step
        dist_sq = np.sum((X[:, None] - centers[None])**2, axis=2)
        labels = np.argmin(dist_sq, axis=1)
        new_centers = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
                                for k in range(n_clusters)])
        # convergence check
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers
    inertia = np.sum((X - centers[labels])**2)
    return float(inertia)

def grade(sol, fx) -> dict:
    # fixed dataset and seeds for reproducibility
    X = np.array([[0., 0.],
                  [1., 0.],
                  [0., 2.],
                  [3., 4.]])
    n_clusters = 2
    seeds = [42, 123]
    try:
        random_inertias, kpp_inertias = sol.compare_inertia(X, n_clusters, seeds)
    except Exception as e:
        return {"rel_err": float("inf")}
    # compute reference inertias
    ref_random = np.array([_kmeans(X, n_clusters, "random", s) for s in seeds], dtype=np.float64)
    ref_kpp   = np.array([_kmeans(X, n_clusters, "kpp", s)     for s in seeds], dtype=np.float64)
    # compute relative error
    def rel_err(a, b):
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        return float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))
    err_random = rel_err(random_inertias, ref_random)
    err_kpp    = rel_err(kpp_inertias, ref_kpp)
    # overall error is the maximum of the two
    return {"rel_err": max(err_random, err_kpp)}
