import numpy as np


def _oracle_kmeans_labels(X, k, centers, iterations):
    X = np.asarray(X, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64).copy()

    for _ in range(iterations):
        distances = np.sum((X[:, None, :] - c[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)

        new_c = c.copy()
        point_distances = distances[np.arange(len(X)), labels]

        for j in range(k):
            mask = labels == j
            if np.any(mask):
                new_c[j] = np.mean(X[mask], axis=0)
            else:
                idx = int(np.argmax(point_distances))
                new_c[j] = X[idx]

        c = new_c

    return labels.astype(int)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[0.0, 0.0], [0.2, 0.1], [5.0, 5.0], [5.2, 5.1]]),
            4,
            np.array([[0.0, 0.0], [5.0, 5.0], [10.0, 10.0], [20.0, 20.0]]),
            4,
        ),
        (
            np.array([[1.0, 1.0], [1.1, 1.0], [1.2, 0.9], [8.0, 8.0]]),
            3,
            np.array([[1.0, 1.0], [1.5, 1.5], [30.0, 30.0]]),
            3,
        ),
    ]

    for X, k, centers, iterations in cases:
        expected = _oracle_kmeans_labels(X, k, centers, iterations)
        try:
            got = np.asarray(sol.kmeans_labels(X, k, centers, iterations))
        except Exception:
            return {"exact_match": 0.0}
        if got.dtype.kind not in "iu":
            return {"exact_match": 0.0}
        if not np.array_equal(got, expected):
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}
