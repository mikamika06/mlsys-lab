import numpy as np

def grade(sol, fx) -> dict:
    ok = 1.0
    for _ in range(5):
        n = np.random.randint(5, 15)
        d = np.random.randint(2, 8)
        X = np.random.randn(n, d).astype(np.float64)
        k = np.random.randint(1, n)  # exclude self
        try:
            got = sol.l2_vs_cosine_neighbor_set_divergence(X, k)
            if not isinstance(got, np.ndarray):
                ok = 0.0
                break
            got = got.astype(bool)
        except Exception:
            ok = 0.0
            break

        # Reference implementation
        norms_sq = np.sum(X**2, axis=1)
        D = norms_sq[:, None] + norms_sq[None, :] - 2 * X.dot(X.T)
        np.fill_diagonal(D, np.inf)
        l2_neighbors = np.argsort(D, axis=1)[:, :k]

        X_norms = np.linalg.norm(X, axis=1)
        X_normalized = X / X_norms[:, None]
        cos_sim = X_normalized @ X_normalized.T
        np.fill_diagonal(cos_sim, -np.inf)
        cos_neighbors = np.argsort(-cos_sim, axis=1)[:, :k]

        ref = np.array([set(l2_neighbors[i]) != set(cos_neighbors[i])
                        for i in range(n)], dtype=bool)

        if not np.array_equal(got, ref):
            ok = 0.0
            break

    return {"exact_match": ok}
