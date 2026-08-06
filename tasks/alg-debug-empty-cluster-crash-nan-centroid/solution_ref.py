def kmeans_labels(
    X: list[list[float]],
    k: int,
    centers: list[list[float]],
    iterations: int
) -> list[int]:
    n = len(X)
    d = len(X[0])
    c = [list(center) for center in centers]

    for _ in range(iterations):
        point_distances = []
        labels = []

        for x in X:
            dists = []
            for ci in c:
                dist = sum((xr - cr) ** 2 for xr, cr in zip(x, ci))
                dists.append(dist)

            min_dist = dists[0]
            best_j = 0
            for j, dist in enumerate(dists):
                if dist < min_dist:
                    min_dist = dist
                    best_j = j

            labels.append(best_j)
            point_distances.append(min_dist)

        new_c = [list(ci) for ci in c]
        cluster_sums = [[0.0] * d for _ in range(k)]
        cluster_counts = [0] * k

        for x, lbl in zip(X, labels):
            cluster_counts[lbl] += 1
            for r in range(d):
                cluster_sums[lbl][r] += x[r]

        for j in range(k):
            if cluster_counts[j] > 0:
                new_c[j] = [cluster_sums[j][r] / cluster_counts[j] for r in range(d)]
            else:
                max_idx = 0
                max_val = point_distances[0]
                for idx, val in enumerate(point_distances):
                    if val > max_val:
                        max_val = val
                        max_idx = idx
                new_c[j] = list(X[max_idx])

        c = new_c

    return labels
