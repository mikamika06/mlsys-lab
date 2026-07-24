import numpy as np


def _channel_rel_err(ref, got):
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    num = np.linalg.norm(got - ref, axis=0)
    den = np.linalg.norm(ref, axis=0) + 1e-12
    return float(np.mean(num / den))


def _oracle_projection(X, k):
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:k].T


def grade(sol, fx) -> dict:
    cases = [
        (np.array([[2.0, 0.0], [0.0, 1.0], [-2.0, 0.0]], dtype=np.float64), 1),
        (np.array([[1.0, 3.0, 2.0], [2.0, 0.0, -1.0], [4.0, 1.0, 5.0], [-2.0, 2.0, 0.0]], dtype=np.float64), 2),
        (np.array([[0.5, -1.0, 2.0, 3.0], [1.5, 2.0, -2.0, 0.0], [-3.0, 1.0, 1.0, 2.0], [4.0, -2.0, 0.5, -1.0], [2.0, 3.0, 1.0, -4.0]], dtype=np.float64), 3),
    ]

    worst = 0.0
    for X, k in cases:
        try:
            got = sol.pca_projection(X.copy(), k)
        except Exception:
            return {"channel_rel_err": float("inf")}
        ref = _oracle_projection(X, k)
        if np.shape(got) != np.shape(ref):
            return {"channel_rel_err": float("inf")}
        worst = max(worst, _channel_rel_err(ref, got))
    return {"channel_rel_err": worst}
