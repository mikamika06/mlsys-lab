import numpy as np


def kmeans_labels(X, k, centers, iterations):
    X = np.asarray(X, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64).copy()

    for _ in range(iterations):
        distances = np.sum((X[:, None, :] - c[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        new_c = c.copy()
        assigned_distances = distances[np.arange(len(X)), labels]

        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_c[j] = np.mean(X[mask], axis=0)
            else:
                idx = np.argmax(assigned_distances)
                new_c[j] = X[idx]

        c = new_c

    return labels.astype(int)
