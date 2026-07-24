import numpy as np


def _oracle_rank(A, tol):
    return int(np.linalg.matrix_rank(np.asarray(A, dtype=np.float64), tol=tol))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 2.0],
            ]),
            1e-10,
        ),
        (
            np.array([
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [4.0, 5.0, 6.0],
            ]),
            1e-12,
        ),
        (
            np.diag([5.0, 2.0, 0.5, 0.0]),
            1e-9,
        ),
        (
            np.array([
                [3.0, -1.0],
                [0.0, 4.0],
                [2.0, 1.0],
            ]),
            1e-8,
        ),
        (
            np.array([
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]),
            1e-9,
        ),
        (
            np.diag([3.0, 1.0, 0.25]),
            1.0,
        ),
    ]

    ok = 1.0
    for A, tol in cases:
        try:
            got = int(sol.svd_rank(A.copy(), tol))
        except Exception:
            ok = 0.0
            break
        ref = _oracle_rank(A, tol)
        if got != ref:
            ok = 0.0
            break

    return {"exact_match": ok}
