import numpy as np


def loo_knn_predict(X: np.ndarray, y: np.ndarray, k: int, n_classes: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    n_samples = X.shape[0]
    n_features = X.shape[1]

    predictions = []

    for i in range(n_samples):
        distances = []
        for j in range(n_samples):
            if i == j:
                distances.append(float('inf'))
            else:
                dist = 0.0
                for d in range(n_features):
                    diff = X[i, d] - X[j, d]
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

    return np.asarray(predictions, dtype=np.int64)
