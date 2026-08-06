import numpy as np


def kmeans_labels(X, k, centers, iterations):
    X = np.asarray(X, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64).copy()

    N = X.shape[0]
    D = X.shape[1]

    labels = [0] * N

    for _ in range(iterations):
        assigned_distances = []

        for i in range(N):
            min_dist = float("inf")
            best_j = 0
            for j in range(k):
                dist = 0.0
                for d in range(D):
                    diff = X[i, d] - c[j, d]
                    dist += diff * diff
                if dist < min_dist:
                    min_dist = dist
                    best_j = j
            labels[i] = best_j
            assigned_distances.append(min_dist)

        new_c = np.empty((k, D), dtype=np.float64)

        for j in range(k):
            count = 0
            for i in range(N):
                if labels[i] == j:
                    count += 1

            if count > 0:
                for d in range(D):
                    val_sum = 0.0
                    for i in range(N):
                        if labels[i] == j:
                            val_sum += X[i, d]
                    new_c[j, d] = val_sum / count
            else:
                max_dist = assigned_distances[0]
                max_idx = 0
                for i in range(1, N):
                    if assigned_distances[i] > max_dist:
                        max_dist = assigned_distances[i]
                        max_idx = i
                for d in range(D):
                    new_c[j, d] = X[max_idx, d]

        c = new_c

    return np.array(labels, dtype=int)
