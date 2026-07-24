import numpy as np

def knn_predict(X_tr: np.ndarray, y_tr: np.ndarray, q: np.ndarray) -> int:
    """
    Return the label of the training point closest to `q` using Euclidean distance.
    The implementation is fully vectorised and uses no explicit Python loops.
    """
    # Compute squared Euclidean distances in a vectorised way
    dists = np.linalg.norm(X_tr - q, axis=1)
    idx = int(np.argmin(dists))
    return int(y_tr[idx])
