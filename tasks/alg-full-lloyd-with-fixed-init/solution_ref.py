from __future__ import annotations

def lloyd_fixed_init(
    X: list[list[float]],
    init_centroids: list[list[float]],
    max_iter: int = 300,
    tol: float = 1e-4
) -> tuple[list[int], int]:
    n_samples = len(X)
    n_features = len(X[0])
    k_clusters = len(init_centroids)

    centroids = []
    for i in range(k_clusters):
        row = []
        for j in range(n_features):
            row.append(float(init_centroids[i][j]))
        centroids.append(row)

    labels = [0] * n_samples
    labels_prev = None

    for it in range(1, max_iter + 1):
        for i in range(n_samples):
            min_dist_sq = float('inf')
            min_k = 0
            for k in range(k_clusters):
                dist_sq = 0.0
                for d in range(n_features):
                    diff = float(X[i][d]) - centroids[k][d]
                    dist_sq += diff * diff
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    min_k = k
            labels[i] = min_k

        if labels_prev is not None:
            same = True
            for i in range(n_samples):
                if labels[i] != labels_prev[i]:
                    same = False
                    break
            if same:
                return list(labels), it - 1

        if labels_prev is None:
            labels_prev = [0] * n_samples
        for i in range(n_samples):
            labels_prev[i] = labels[i]

        for k in range(k_clusters):
            count = 0
            for i in range(n_samples):
                if labels[i] == k:
                    count += 1
            if count > 0:
                for d in range(n_features):
                    sum_val = 0.0
                    for i in range(n_samples):
                        if labels[i] == k:
                            sum_val += float(X[i][d])
                    centroids[k][d] = sum_val / count

    return list(labels), max_iter
