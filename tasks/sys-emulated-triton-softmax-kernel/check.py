import numpy as np


def _oracle_softmax(X):
    X = np.asarray(X, dtype=np.float64)
    m = np.max(X, axis=1, keepdims=True)
    e = np.exp(X - m)
    return e / np.sum(e, axis=1, keepdims=True)


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float64),
        rng.normal(size=(8, 17)).astype(np.float64),
        rng.normal(loc=20.0, scale=5.0, size=(5, 33)).astype(np.float64),
        np.array([[1000.0, 999.0, 998.0], [-1000.0, -1001.0, -999.0]], dtype=np.float64),
        np.array([[2000.0, 1999.0, 1998.0, 1997.0]], dtype=np.float64),
    ]

    worst = 0.0
    for X in cases:
        ref = _oracle_softmax(X)
        try:
            got = np.asarray(sol.softmax_kernel(X), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        if got.shape != ref.shape or not np.all(np.isfinite(got)):
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(got - ref)))
        if not np.isfinite(err):
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)

    return {"max_abs_err": worst}
