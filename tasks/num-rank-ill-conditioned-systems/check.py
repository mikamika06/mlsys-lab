import numpy as np


def _oracle_order(matrices):
    conds = [float(np.linalg.cond(np.asarray(m, dtype=np.float64))) for m in matrices]
    return sorted(range(len(matrices)), key=lambda i: conds[i])


def grade(sol, fx) -> dict:
    cases = [
        [
            np.eye(3),
            np.array([[1.0, 1.0], [1.0, 1.000001]]),
            np.array([[3.0, 0.0], [0.0, 0.25]]),
            np.array([[1.0, 0.2], [0.2, 1.0]]),
        ],
        [
            np.array([[1.0, 0.0], [0.0, 1e-8]]),
            np.array([[5.0, 0.0], [0.0, 5.0]]),
            np.array([[1.0, 2.0], [0.0, 1.0]]),
            np.array([[4.0, 1.0], [1.0, 4.0]]),
        ],
        [
            np.array([[2.0, -1.0], [0.0, 2.0]]),
            np.array([[1.0, 1.0], [1.0, 1.0000001]]),
            np.array([[10.0, 0.0], [0.0, 0.1]]),
        ],
    ]

    ok = 1.0
    for mats in cases:
        try:
            got = list(sol.rank_by_condition([m.copy() for m in mats]))
        except Exception:
            ok = 0.0
            break
        if got != _oracle_order(mats):
            ok = 0.0
            break
    return {"exact_match": ok}
