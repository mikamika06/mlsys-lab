import numpy as np

def predict_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    k: int
) -> np.ndarray:
    """Correct implementation of k‑Nearest Neighbours with smallest‑label tie‑break."""
    # Compute pairwise Euclidean distances (test × train)
    dists = np.linalg.norm(X_train[None] - X_test[:, None], axis=2)
    # Indices of the k nearest neighbours for each test point
    idxs = np.argsort(dists, axis=1)[:, :k]
    preds = []
    max_label = int(y_train.max())
    for i in range(X_test.shape[0]):
        labels = y_train[idxs[i]]
        counts = np.bincount(labels, minlength=max_label + 1)
        # np.argmax returns the first index of the maximum value,
        # which is exactly the smallest label when there is a tie.
        preds.append(int(np.argmax(counts)))
    return np.array(preds)
