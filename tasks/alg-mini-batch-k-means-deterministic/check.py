import numpy as np
from mlsys import scorers

def _reference(X, k, batch_size, n_iter, seed):
    rng = np.random.default_rng(seed)
    centroids = X[:k].astype(np.float64).copy()
    for it in range(1, n_iter + 1):
        idx = rng.choice(len(X), size=batch_size, replace=True)
        batch = X[idx]
        dists = np.sum((batch[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)

        new_centroids = centroids.copy()
        for i in range(k):
            mask = (labels == i)
            if np.any(mask):
                new_centroids[i] = np.mean(batch[mask], axis=0)

        centroids = (centroids * (it - 1) + new_centroids) / it
    return centroids

def grade(sol, fx) -> dict:
    tests = [
        (np.random.default_rng(0).random((20, 3)), 4, 5, 15, 123),
        (np.random.default_rng(1).random((50, 2)), 3, 10, 20, 42),
        (np.random.default_rng(2).random((100, 5)), 5, 8, 25, 7),
    ]

    max_err = 0.0
    for X, k, bs, iters, seed in tests:
        try:
            got = sol.mini_batch_kmeans(X, k, bs, iters, seed)
            ref = _reference(X, k, bs, iters, seed)
        except Exception:
            return {"rel_err": float("inf")}
        err = scorers.rel_err(ref, got)
        if err > max_err:
            max_err = err
    return {"rel_err": max_err}
