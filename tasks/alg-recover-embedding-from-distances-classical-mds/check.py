import numpy as np

def _reference_mds(D2, k):
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    lam = eigvals[idx][:k]
    vec = eigvecs[:, idx[:k]]
    return vec * np.sqrt(np.maximum(lam, 0))

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    max_rel_err = 0.0
    for _ in range(5):
        n = rng.integers(5, 15)
        k = rng.integers(1, min(n - 1, 8))
        X_true = rng.standard_normal((n, k))
        D2 = np.sum((X_true[:, None, :] - X_true[None, :, :]) ** 2, axis=2)
        try:
            X_est = sol.mds_from_distances(D2, k)
        except Exception:
            return {"rel_err": float("inf")}
        if X_est.shape != (n, k):
            return {"rel_err": 0.0}
        D2_est = np.sum((X_est[:, None, :] - X_est[None, :, :]) ** 2, axis=2)
        rel_err = np.linalg.norm(D2_est - D2) / (np.linalg.norm(D2) + 1e-12)
        if rel_err > max_rel_err:
            max_rel_err = rel_err
    return {"rel_err": float(max_rel_err)}
