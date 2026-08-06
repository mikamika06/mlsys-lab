import numpy as np
from mlsys.scorers import rel_err

def _reference_knn(X_train, y_train, X_query, k):
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

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(5):
        n_train = rng.integers(10, 50)
        n_query = rng.integers(5, 20)
        d = rng.integers(3, 8)
        X_train = rng.standard_normal((n_train, d))
        y_train = rng.standard_normal(n_train)
        X_query = rng.standard_normal((n_query, d))
        k = rng.integers(1, n_train + 1)
        try:
            ref = _reference_knn(X_train, y_train, X_query, k)
            cand = sol.knn_regression_average(X_train.tolist(), y_train.tolist(), X_query.tolist(), k)
        except Exception:
            return {"rel_err": float("inf")}
        if not isinstance(cand, list) or len(cand) != ref.shape[0]:
            return {"rel_err": float("inf")}
        cand_arr = np.asarray(cand, dtype=np.float64)
        err = rel_err(ref, cand_arr)
        if err > max_err:
            max_err = err
    return {"rel_err": max_err}
