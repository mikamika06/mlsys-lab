import numpy as np


def _oracle_reconstruction(A):
    q, r = np.linalg.qr(A, mode="complete")
    return q @ r


def grade(sol, fx) -> dict:
    cases = [
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        np.array([
            [2.0, -1.0, 3.0],
            [4.0, 5.0, 6.0],
            [-2.0, 1.0, 0.5],
            [3.0, 2.0, -4.0],
        ]),
        np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]),
    ]

    worst = 0.0
    for a in cases:
        ref = _oracle_reconstruction(a)
        m, n = a.shape
        try:
            q, r = sol.householder_qr_reconstruct(a.copy())
            q = np.asarray(q, dtype=np.float64)
            r = np.asarray(r, dtype=np.float64)

            if q.shape != (m, m) or r.shape != (m, n):
                return {"max_abs_err": float("inf")}

            got = q @ r
            err = float(np.max(np.abs(got - ref)))
        except Exception:
            return {"max_abs_err": float("inf")}

        worst = max(worst, err)

    return {"max_abs_err": worst}
