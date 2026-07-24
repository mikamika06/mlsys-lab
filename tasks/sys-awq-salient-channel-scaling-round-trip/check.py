import numpy as np


def _oracle_awq(W, X):
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    scale = np.max(np.abs(W), axis=0, keepdims=True) / 7.0
    scale = np.where(scale == 0, 1.0, scale)
    q = np.round(W / scale)
    q = np.clip(q, -8, 7)
    W_hat = q * scale
    return W_hat @ X


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(123)
    cases = []
    for seed in range(4):
        local = np.random.default_rng(seed)
        W = local.normal(size=(16, 32))
        W[:, [2, 7, 13]] *= 10.0
        X = local.normal(size=(32, 24))
        cases.append((W, X))

    worst = 0.0
    for W, X in cases:
        try:
            got = np.asarray(sol.awq_matmul(W, X), dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}
        ref = W @ X
        err = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
        worst = max(worst, float(err))

    return {"rel_err": worst}
