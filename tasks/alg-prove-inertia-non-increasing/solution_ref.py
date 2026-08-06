import numpy as np

def inertia_sequence(X: np.ndarray, n_clusters: int, max_iter: int = 10) -> list[float]:
    rng = np.random.default_rng(0)
    n_samples = len(X)
    n_features = X.shape[1]
    indices = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = X[indices]
    seq = []

    for _ in range(max_iter):
        labels = [0] * n_samples
        for i in range(n_samples):
            min_d2 = 0.0
            for d in range(n_features):
                diff = X[i, d] - centroids[0, d]
                min_d2 += diff * diff
            best_k = 0

            for k in range(1, n_clusters):
                d2 = 0.0
                for d in range(n_features):
                    diff = X[i, d] - centroids[k, d]
                    d2 += diff * diff
                if d2 < min_d2:
                    min_d2 = d2
                    best_k = k
            labels[i] = best_k

        inertia = 0.0
        for i in range(n_samples):
            k = labels[i]
            for d in range(n_features):
                diff = X[i, d] - centroids[k, d]
                inertia += diff * diff
        seq.append(float(inertia))

        counts = [0] * n_clusters
        for i in range(n_samples):
            counts[labels[i]] += 1

        sums = [[0.0] * n_features for _ in range(n_clusters)]
        for d in range(n_features):
            for i in range(n_samples):
                sums[labels[i]][d] += float(X[i, d])

        new_centroids = np.empty((n_clusters, n_features), dtype=X.dtype)
        for k in range(n_clusters):
            if counts[k] == 0:
                for d in range(n_features):
                    new_centroids[k, d] = centroids[k, d]
            else:
                for d in range(n_features):
                    new_centroids[k, d] = sums[k][d] / counts[k]

        all_close = True
        for k in range(n_clusters):
            for d in range(n_features):
                diff_val = new_centroids[k, d] - centroids[k, d]
                if diff_val < 0:
                    diff_val = -diff_val
                b_val = centroids[k, d]
                if b_val < 0:
                    b_val = -b_val
                if diff_val > 1e-8 + 1e-5 * b_val:
                    all_close = False
                    break
            if not all_close:
                break

        if all_close:
            break
        centroids = new_centroids

    return seq
