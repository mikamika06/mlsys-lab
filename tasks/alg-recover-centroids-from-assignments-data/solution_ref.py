def recover_centroids(X: list[list[float]], labels: list[int]) -> list[list[float]]:
    max_lbl = labels[0]
    for lbl in labels:
        if lbl > max_lbl:
            max_lbl = lbl
    k = max_lbl + 1

    N = len(X)
    D = len(X[0])
    centroids = []
    for i in range(k):
        sums = [0.0] * D
        count = 0
        for n in range(N):
            if labels[n] == i:
                count += 1
                for d in range(D):
                    sums[d] += X[n][d]
        if count > 0:
            centroids.append([sums[d] / count for d in range(D)])
        else:
            centroids.append([0.0] * D)
    return centroids
