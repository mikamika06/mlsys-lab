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
            dists = []
            for i in range(n_samples):
                min_d = 0.0
                for c in range(len(centers)):
                    d_sq = 0.0
                    for f in range(n_features):
                        diff = float(X[i, f]) - float(centers[c, f])
                        d_sq += diff * diff
                    if c == 0 or d_sq < min_d:
                        min_d = d_sq
                dists.append(min_d)
            dists_sum = 0.0
            for d in dists:
                dists_sum += d
            probs = np.array([d / dists_sum for d in dists], dtype=np.float64)
            idx.append(rng.choice(n_samples, p=probs))
            centers = X[idx]
    else:
        raise ValueError("init must be 'random' or 'kpp'")
    for _ in range(300):
        labels = []
        for i in range(n_samples):
            best_k = 0
            min_dist = 0.0
            for k in range(n_clusters):
                d_sq = 0.0
                for f in range(n_features):
                    diff = float(X[i, f]) - float(centers[k, f])
                    d_sq += diff * diff
                if k == 0 or d_sq < min_dist:
                    min_dist = d_sq
                    best_k = k
            labels.append(best_k)
        new_centers_list = []
        for k in range(n_clusters):
            count = 0
            for i in range(n_samples):
                if labels[i] == k:
                    count += 1
            if count > 0:
                center_k = []
                for f in range(n_features):
                    sum_f = 0.0
                    for i in range(n_samples):
                        if labels[i] == k:
                            sum_f += float(X[i, f])
                    center_k.append(sum_f / count)
                new_centers_list.append(center_k)
            else:
                new_centers_list.append([float(centers[k, f]) for f in range(n_features)])
        new_centers = np.array(new_centers_list, dtype=np.float64)
        converged = True
        for k in range(n_clusters):
            for f in range(n_features):
                diff = centers[k, f] - new_centers[k, f]
                if diff < 0:
                    diff = -diff
                b_val = new_centers[k, f]
                if b_val < 0:
                    b_val = -b_val
                if diff > 1e-4 + 1e-5 * b_val:
                    converged = False
                    break
            if not converged:
                break
        if converged:
            break
        centers = new_centers
    inertia = 0.0
    for i in range(n_samples):
        k = labels[i]
        for f in range(n_features):
            diff = float(X[i, f]) - float(centers[k, f])
            inertia += diff * diff
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
