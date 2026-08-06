import numpy as np
import random
from mlsys import scorers

def _reference(X, k, batch_size, n_iter, seed):
    rng = random.Random(seed)
    n_samples = len(X)
    centroids = np.array([X[i] for i in range(k)], dtype=float)
    for it in range(1, n_iter + 1):
        idx = [rng.randrange(n_samples) for _ in range(batch_size)]
        batch = np.array([X[i] for i in idx], dtype=float)
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
        (np.random.default_rng(0).random((20, 3)).tolist(), 4, 5, 15, 123),
        (np.random.default_rng(1).random((50, 2)).tolist(), 3, 10, 20, 42),
        (np.random.default_rng(2).random((100, 5)).tolist(), 5, 8, 25, 7),
    ]

    max_err = 0.0
    for X, k, bs, iters, seed in tests:
        try:
            got = sol.mini_batch_kmeans(X, k, bs, iters, seed)
            ref = _reference(X, k, bs, iters, seed)
            got_arr = np.array(got, dtype=float)
        except Exception:
            return {"rel_err": float("inf")}
        err = scorers.rel_err(ref, got_arr)
        if err > max_err:
            max_err = err
    return {"rel_err": max_err}
