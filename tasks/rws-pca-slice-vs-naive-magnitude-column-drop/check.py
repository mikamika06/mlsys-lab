import numpy as np


def _pca_mse(X, k):
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vk = Vt[:k].T
    Z = Xc @ Vk
    Xhat = Z @ Vk.T + mean
    return float(np.mean((X - Xhat) ** 2))


def _naive_mse(X, k):
    norms = np.linalg.norm(X, axis=0)
    order = np.argsort(-norms, kind="stable")
    keep = order[:k]
    Xhat = np.zeros_like(X)
    Xhat[:, keep] = X[:, keep]
    return float(np.mean((X - Xhat) ** 2))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    worst_abs = 0.0
    ordering_ok = 1.0

    for _ in range(6):
        n = int(rng.integers(20, 80))
        d = int(rng.integers(6, 20))
        k = int(rng.integers(1, d))
        X = rng.normal(size=(n, d)) * rng.uniform(0.3, 3.0, size=(1, d))

        exp_pca = _pca_mse(X, k)
        exp_naive = _naive_mse(X, k)
        assert exp_pca <= exp_naive + 1e-9, "Eckart-Young: PCA must be optimal rank-k"

        try:
            got_pca, got_naive = sol.pca_vs_naive_mse(X.copy(), k)
            got_pca = float(got_pca)
            got_naive = float(got_naive)
        except Exception:
            return {"max_abs_err": float("inf"), "ordering_ok": 0.0}

        worst_abs = max(worst_abs, abs(got_pca - exp_pca), abs(got_naive - exp_naive))
        if not (got_pca <= got_naive + 1e-9):
            ordering_ok = 0.0

    return {"max_abs_err": worst_abs, "ordering_ok": ordering_ok}
