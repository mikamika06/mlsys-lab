import numpy as np

def _reference_assign(X, C):
    X_arr = np.array(X)
    C_arr = np.array(C)
    diff = X_arr[:, None, :] - C_arr[None, :, :]
    dists = np.sum(diff**2, axis=2)
    return np.argmin(dists, axis=1).tolist()

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = int(rng.integers(3, 10))
        d = int(rng.integers(2, 6))
        k = int(rng.integers(2, 5))
        X_np = rng.standard_normal((n, d))
        centroids_np = rng.standard_normal((k, d))
        X = X_np.tolist()
        centroids = centroids_np.tolist()
        try:
            got = sol.assign_clusters(X, centroids)
        except Exception:
            return {"exact_match": 0.0}
        ref = _reference_assign(X, centroids)
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
