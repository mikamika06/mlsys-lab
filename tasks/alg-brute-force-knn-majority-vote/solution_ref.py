import numpy as np

def knn_majority_vote(Xtr: np.ndarray,
                      ytr: np.ndarray,
                      Xte: np.ndarray,
                      k: int) -> np.ndarray:
    """
    Brute‑force kNN classifier with majority vote and deterministic tie‑break.
    Parameters
    ----------
    Xtr : (n_train, d) array of training samples
    ytr : (n_train,) integer labels for the training samples
    Xte : (n_test, d) array of test samples to classify
    k   : number of nearest neighbours to consider

    Returns
    -------
    preds : (n_test,) array of predicted integer labels
    """
    # Compute all pairwise squared Euclidean distances
    dists = np.sum((Xte[:, None] - Xtr[None]) ** 2, axis=2)
    # For each test point find indices of the k smallest distances
    knn_idx = np.argpartition(dists, kth=k-1, axis=1)[:, :k]
    preds = []
    for idx in knn_idx:
        labels, counts = np.unique(ytr[idx], return_counts=True)
        max_count = counts.max()
        best_labels = labels[counts == max_count]
        preds.append(best_labels.min())  # deterministic tie‑break
    return np.array(preds, dtype=np.int64)
