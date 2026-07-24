import numpy as np

def knn_grid_labels(
    train_points: np.ndarray,
    train_labels: np.ndarray,
    grid_points: np.ndarray,
    k: int = 3
) -> np.ndarray:
    """
    Vectorised k‑Nearest Neighbours classification on a dense grid.

    Parameters
    ----------
    train_points : (N, d) array of training samples.
    train_labels : (N,) integer labels in {0,…,C-1}.
    grid_points  : (M, d) query points to classify.
    k            : number of neighbours to consider.

    Returns
    -------
    logits : (M, C) one‑hot encoded predictions.
    """
    # Ensure inputs are float64 for consistency
    train_points = np.asarray(train_points, dtype=np.float64)
    grid_points  = np.asarray(grid_points,  dtype=np.float64)

    N, d = train_points.shape
    M     = grid_points.shape[0]
    C     = int(train_labels.max()) + 1

    # Compute squared Euclidean distances: (M, N)
    dists = ((grid_points[:, None, :] - train_points[None, :, :]) ** 2).sum(axis=2)

    # Indices of the k nearest neighbours for each query point
    idx = np.argpartition(dists, kth=k-1, axis=1)[:, :k]   # shape (M, k)

    # Gather neighbour labels: (M, k)
    neigh_labels = train_labels[idx]

    # Majority vote per row; ties resolved to the smallest label
    counts = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=C),
        axis=1,
        arr=neigh_labels
    )
    preds = counts.argmax(axis=1)   # shape (M,)

    # One‑hot encode predictions
    logits = np.eye(C)[preds].astype(np.float64)
    return logits
