import numpy as np
from mlsys.scorers import argmax_agreement

def _reference_knn(Xtr, ytr, Xte, k):
    # Compute all pairwise squared Euclidean distances
    dists = np.sum((Xte[:, None] - Xtr[None])**2, axis=2)
    # For each test point find indices of the k smallest distances
    knn_idx = np.argpartition(dists, kth=k-1, axis=1)[:, :k]
    # Gather labels and perform majority vote with deterministic tie‑break
    preds = []
    for idx in knn_idx:
        labels, counts = np.unique(ytr[idx], return_counts=True)
        max_count = counts.max()
        best_labels = labels[counts == max_count]
        preds.append(best_labels.min())
    return np.array(preds, dtype=np.int64)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    n_tr, d = 200, 5
    Xtr = rng.standard_normal((n_tr, d))
    ytr = rng.integers(low=0, high=4, size=n_tr)
    n_te = 50
    Xte = rng.standard_normal((n_te, d))
    k = 3

    ref_pred = _reference_knn(Xtr, ytr, Xte, k)

    try:
        cand_pred = sol.knn_majority_vote(Xtr, ytr, Xte, k)
    except Exception:
        return {"argmax_agreement": 0.0}

    if not isinstance(cand_pred, np.ndarray):
        return {"argmax_agreement": 0.0}
    if cand_pred.shape != (n_te,):
        return {"argmax_agreement": 0.0}

    score = argmax_agreement(ref_pred, cand_pred)
    return {"argmax_agreement": float(score)}
