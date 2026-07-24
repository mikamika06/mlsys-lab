import numpy as np


def _oracle(L, x):
    A = L @ L.T
    updated = A + np.outer(x, x)
    return np.linalg.cholesky(updated).astype(np.float64)


def grade(sol, fx) -> dict:
    cases = []
    for n in [2, 3, 5, 8]:
        rng = np.random.default_rng(100 + n)
        M = rng.normal(size=(n, n))
        A = M @ M.T + np.eye(n) * 0.5
        L = np.linalg.cholesky(A).astype(np.float64)
        x = rng.normal(size=n).astype(np.float64)
        cases.append((L, x))

    max_err = 0.0
    for L, x in cases:
        try:
            got = np.asarray(sol.rank1_cholesky_update(L.copy(), x.copy()), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = _oracle(L, x)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        max_err = max(max_err, float(np.max(np.abs(got - ref))))

    return {"max_abs_err": max_err}
