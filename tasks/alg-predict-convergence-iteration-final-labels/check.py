import numpy as np

def _oracle_kmeans(X, k, max_iter=300, tol=1e-4):
    rng = np.random.default_rng(0)
    centroids = X[rng.choice(len(X), size=k, replace=False)]
    labels = np.empty(len(X), dtype=int)
    for it in range(max_iter):
        # Assignment step
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)
        new_labels = np.argmin(dists, axis=1)
        # Update step
        new_centroids = np.array([
            X[new_labels == j].mean(axis=0) if np.any(new_labels == j) else centroids[j]
            for j in range(k)
        ])
        shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
        centroids = new_centroids
        labels = new_labels
        if shift < tol:
            return it + 1, labels.copy()
    return max_iter, labels.copy()

def grade(sol, fx) -> dict:
    cases = [
        (np.array([[0, 0], [1, 0], [0, 2]]), 2),
        (np.random.default_rng(42).random((10, 3)), 3),
        (np.random.default_rng(7).random((50, 5)), 4),
    ]
    ok = 1.0
    for X, k in cases:
        try:
            got_iters, got_labels = sol.predict_kmeans_convergence(X, k)
        except Exception:
            return {"exact_match": 0.0}
        ref_iters, ref_labels = _oracle_kmeans(X, k)
        if got_iters != ref_iters or not np.array_equal(got_labels, ref_labels):
            ok = 0.0
            break
    return {"exact_match": ok}
