import numpy as np


def loo_knn_predict(X: np.ndarray, y: np.ndarray, k: int, n_classes: int) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    distances = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)

    neighbors = np.argsort(distances, axis=1)[:, :k]
    predictions = np.empty(X.shape[0], dtype=np.int64)

    for i, row in enumerate(neighbors):
        counts = np.bincount(y[row], minlength=n_classes)
        predictions[i] = np.argmax(counts)

    return predictions
