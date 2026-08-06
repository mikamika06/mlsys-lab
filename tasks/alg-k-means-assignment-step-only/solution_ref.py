def assign_clusters(X: list[list[float]], centroids: list[list[float]]) -> list[int]:
    n = len(X)
    d = len(X[0])
    k = len(centroids)
    labels = [0] * n
    for i in range(n):
        best_j = 0
        best_dist = float("inf")
        for j in range(k):
            dist = 0.0
            for m in range(d):
                diff = X[i][m] - centroids[j][m]
                dist += diff * diff
            if dist < best_dist:
                best_dist = dist
                best_j = j
        labels[i] = best_j
    return labels
