import numpy as np


def kmeans_labels(X, k, centers, iterations):
    # TODO: empty clusters are incorrectly updated with the mean of an empty slice,
    # creating NaN centroids and incorrect later assignments.
    X = np.asarray(X, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64).copy()

    for _ in range(iterations):
        distances = np.sum((X[:, None, :] - c[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        new_c = c.copy()
        for j in range(k):
            mask = labels == j
            new_c[j] = np.mean(X[mask], axis=0)

        c = new_c

    return labels.astype(int)
