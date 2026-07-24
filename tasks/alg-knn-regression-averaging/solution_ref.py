import numpy as np

def knn_regression_average(X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_query: np.ndarray,
                           k: int) -> np.ndarray:
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_query = np.asarray(X_query, dtype=np.float64)
    n_train = X_train.shape[0]
    if k > n_train:
        raise ValueError("k cannot exceed number of training samples")
    diff = X_train[:, None, :] - X_query[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)  # shape (n_train, n_query)
    idx = np.argpartition(dist_sq, kth=k-1, axis=0)[:k, :]
    neigh_vals = y_train[idx]
    preds = np.mean(neigh_vals, axis=0)
    return preds
