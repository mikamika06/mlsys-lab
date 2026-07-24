import numpy as np


def _oracle_householder(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.shape[0]
    norm = np.linalg.norm(x)
    if norm == 0:
        return np.eye(m, dtype=np.float64)
    sign = -1.0 if x[0] >= 0 else 1.0
    alpha = sign * norm
    v = x.copy()
    v[0] -= alpha
    beta = np.dot(v, v)
    if beta == 0:
        return np.eye(m, dtype=np.float64)
    H = np.eye(m, dtype=np.float64) - 2.0 * np.outer(v, v) / beta
    return H


def grade(sol, fx) -> dict:
    cases = [
        np.array([10.0, 1e-8, -2e-8], dtype=np.float64),
        np.array([-10.0, 1e-8, 2e-8], dtype=np.float64),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        np.array([1e8, 1.0, -3.0, 5.0], dtype=np.float64),
        np.array([0.0, 2.0, 3.0], dtype=np.float64),
    ]
    worst = 0.0
    try:
        for x in cases:
            H_ref = _oracle_householder(x)
            H = np.asarray(sol.householder_fixed(x.copy()), dtype=np.float64)
            if H.shape != H_ref.shape:
                return {"max_abs_err": 1.0}
            err = float(np.max(np.abs((H @ x) - (H_ref @ x))))
            worst = max(worst, err)
    except Exception:
        worst = 1.0
    return {"max_abs_err": worst}
