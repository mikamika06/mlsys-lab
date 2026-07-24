import numpy as np


def _oracle(A):
    vals = np.linalg.eigvalsh(A)
    return float(vals[-2])


def grade(sol, fx) -> dict:
    cases = [
        np.array([[5.0, 0.0], [0.0, 2.0]]),
        np.array([[6.0, 1.0], [1.0, 3.0]]),
        np.array([
            [8.0, 1.0, 0.5],
            [1.0, 4.0, 0.2],
            [0.5, 0.2, 2.0],
        ]),
        np.array([
            [10.0, 2.0, 0.0, 0.0],
            [2.0, 7.0, 1.0, 0.0],
            [0.0, 1.0, 3.0, 0.5],
            [0.0, 0.0, 0.5, 1.0],
        ]),
    ]

    worst = 0.0
    for A in cases:
        try:
            got = float(sol.second_eigenvalue(A))
        except Exception:
            return {"rel_err": float("inf")}
        ref = _oracle(A)
        err = abs(got - ref) / (abs(ref) + 1e-12)
        worst = max(worst, err)

    return {"rel_err": worst}
