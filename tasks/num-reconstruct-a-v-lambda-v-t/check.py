import numpy as np


def _oracle(A):
    w, V = np.linalg.eigh(A)
    return V @ np.diag(w) @ V.T


def _max_abs_err(a, b):
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def grade(sol, fx) -> dict:
    cases = [
        np.array([[2.0, 1.0], [1.0, 2.0]]),
        np.array([[4.0, -1.0, 2.0], [-1.0, 3.0, 0.5], [2.0, 0.5, 1.0]]),
        np.array([
            [5.0, 2.0, -1.0, 0.0],
            [2.0, 4.0, 1.5, -2.0],
            [-1.0, 1.5, 3.0, 1.0],
            [0.0, -2.0, 1.0, 6.0],
        ]),
    ]
    worst = 0.0
    try:
        for A in cases:
            ref = _oracle(A)
            got = sol.reconstruct_from_eigh(A)
            err = _max_abs_err(ref, got)
            worst = max(worst, err)
    except Exception:
        return {"max_abs_err": float("inf")}
    return {"max_abs_err": worst}
