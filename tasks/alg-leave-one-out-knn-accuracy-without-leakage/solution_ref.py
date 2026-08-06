def loo_knn_predict(X: list[list[float]], y: list[int], k: int, n_classes: int) -> list[int]:
    n_samples = len(X)
    n_features = len(X[0])

    predictions = []

    for i in range(n_samples):
        distances = []
        for j in range(n_samples):
            if i == j:
                distances.append(float('inf'))
            else:
                dist = 0.0
                for d in range(n_features):
                    diff = X[i][d] - X[j][d]
                    dist += diff * diff
                distances.append(dist)

        sorted_indices = sorted(range(n_samples), key=lambda j: distances[j])
        neighbors = sorted_indices[:k]

        counts = [0] * n_classes
        for idx in neighbors:
            label = int(y[idx])
            counts[label] += 1

        max_count = -1
        best_class = 0
        for c in range(n_classes):
            if counts[c] > max_count:
                max_count = counts[c]
                best_class = c

        predictions.append(best_class)

    return predictions
