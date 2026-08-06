def assign_clusters(X: list[list[float]], centroids: list[list[float]]) -> list[int]:
    return [min(range(len(centroids)), key=lambda j: sum((a - b) ** 2 for a, b in zip(x, centroids[j]))) for x in X]
