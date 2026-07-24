import numpy as np

def _reference_assign(X, C):
    diff = X[:, None, :] - C[None, :, :]
    dists = np.sum(diff**2, axis=2)
    return np.argmin(dists, axis=1)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = rng.integers(3, 10)
        d = rng.integers(2, 6)
        k = rng.integers(2, 5)
        X = rng.standard_normal((n, d))
        centroids = rng.standard_normal((k, d))
        try:
            got = sol.assign_clusters(X, centroids)
        except Exception:
            return {"exact_match": 0.0}
        ref = _reference_assign(X, centroids)
        if not np.array_equal(got, ref):
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
