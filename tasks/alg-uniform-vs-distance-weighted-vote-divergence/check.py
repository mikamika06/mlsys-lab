import numpy as np
from mlsys.scorers import argmax_agreement

def _predict(X_train, y_train, X_test, k, weight):
    n_classes = int(y_train.max()) + 1
    # Euclidean distances
    diff = X_test[:, None, :] - X_train[None, :, :]
    dists = np.linalg.norm(diff, axis=2)          # shape (n_test, n_train)
    idx_k = np.argpartition(dists, kth=k-1, axis=1)[:,:k]  # k nearest indices
    neigh_labels = y_train[idx_k]
    neigh_dists  = dists[np.arange(dists.shape[0])[:,None], idx_k]

    if weight == 'uniform':
        counts = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=n_classes), axis=1, arr=neigh_labels)
        preds = np.argmax(counts, axis=1)
    else:  # distance‑weighted
        eps = 1e-12
        weights = 1.0 / (neigh_dists + eps)
        weighted_sums = np.zeros((dists.shape[0], n_classes))
        for cls in range(n_classes):
            mask = neigh_labels == cls
            weighted_sums[:,cls] = np.sum(weights * mask, axis=1)
        preds = np.argmax(weighted_sums, axis=1)

    return preds

def grade(sol, fx) -> dict:
    # deterministic toy dataset
    X_train = np.array([[0., 0.],
                        [1., 0.],
                        [0., 1.],
                        [1., 1.]])
    y_train = np.array([0, 1, 1, 0])

    X_test = np.array([[0.2, 0.2],
                       [0.8, 0.8],
                       [0.5, 0.5]])
    k = 3

    ref_uniform   = _predict(X_train, y_train, X_test, k, 'uniform')
    ref_weighted  = _predict(X_train, y_train, X_test, k, 'distance')

    try:
        uniform_user, weighted_user = sol.knn_vote_divergence(
            X_train, y_train, X_test, k)
    except Exception:
        return {"uniform_agreement": 0.0, "weighted_agreement": 0.0}

    ua = argmax_agreement(ref_uniform, uniform_user)
    wa = argmax_agreement(ref_weighted, weighted_user)

    return {"uniform_agreement": ua, "weighted_agreement": wa}
