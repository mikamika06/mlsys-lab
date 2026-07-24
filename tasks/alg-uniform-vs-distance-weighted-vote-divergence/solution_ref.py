import numpy as np

def knn_vote_divergence(X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test:  np.ndarray,
                        k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Return uniform and distance‑weighted kNN predictions for each point in X_test.
    """
    n_classes = int(y_train.max()) + 1

    # Euclidean distances between all test–train pairs
    diff = X_test[:, None, :] - X_train[None, :, :]
    dists = np.linalg.norm(diff, axis=2)          # (n_test, n_train)

    # indices of the k nearest neighbours for each test point
    idx_k = np.argpartition(dists, kth=k-1, axis=1)[:,:k]  # shape (n_test, k)
    neigh_labels = y_train[idx_k]
    neigh_dists  = dists[np.arange(dists.shape[0])[:,None], idx_k]

    # --- Uniform voting ----------------------------------------------------
    counts = np.apply_along_axis(
        lambda row: np.bincount(row, minlength=n_classes), axis=1, arr=neigh_labels)
    uniform_preds = np.argmax(counts, axis=1)

    # --- Distance‑weighted voting ------------------------------------------
    eps = 1e-12
    weights = 1.0 / (neigh_dists + eps)          # inverse distance
    weighted_sums = np.zeros((dists.shape[0], n_classes))
    for cls in range(n_classes):
        mask = neigh_labels == cls
        weighted_sums[:,cls] = np.sum(weights * mask, axis=1)
    weighted_preds = np.argmax(weighted_sums, axis=1)

    return uniform_preds.astype(np.int64), weighted_preds.astype(np.int64)
