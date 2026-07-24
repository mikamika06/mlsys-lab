import numpy as np

def _reference_lloyd(X, init_centroids, max_iter=300, tol=1e-4):
    centroids = init_centroids.copy()
    labels_prev = None
    for it in range(1, max_iter + 1):
        diff = X[:, None, :] - centroids[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        labels = np.argmin(dist_sq, axis=1)
        if labels_prev is not None and np.array_equal(labels, labels_prev):
            return labels, it - 1
        labels_prev = labels.copy()
        for i in range(centroids.shape[0]):
            mask = (labels == i)
            if np.any(mask):
                centroids[i] = X[mask].mean(axis=0)
    return labels, max_iter

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    ok = 1.0
    for _ in range(5):
        n = rng.integers(10, 50)
        d = rng.integers(2, 6)
        k = rng.integers(2, min(n // 2, 8))
        X = rng.standard_normal((n, d))
        init_centroids = rng.standard_normal((k, d))
        try:
            cand_labels, cand_iter = sol.lloyd_fixed_init(X, init_centroids)
        except Exception:
            ok = 0.0
            break
        ref_labels, _ = _reference_lloyd(X, init_centroids)
        if not isinstance(cand_labels, np.ndarray) or cand_labels.shape != (n,) or cand_labels.dtype.kind != 'i':
            ok = 0.0
            break
        if not np.array_equal(cand_labels, ref_labels):
            ok = 0.0
            break
        if not isinstance(cand_iter, int) or cand_iter < 1:
            ok = 0.0
            break
    return {"exact_match": ok}
