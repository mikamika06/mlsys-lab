import numpy as np


def _oracle(A, b):
    return np.linalg.solve(np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64))


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(7)

    cases = [
        (
            np.array([[3.0, 1.0], [1.0, 2.0]]),
            np.array([9.0, 8.0]),
        ),
        (
            np.array([
                [0.0, 2.0, 1.0],
                [1.0, -1.0, 2.0],
                [2.0, 1.0, 1.0],
            ]),
            np.array([4.0, 1.0, 7.0]),
        ),
        (
            np.array([
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 17.0],
            ]),
            np.array([1.0, 2.0, 3.0, 4.0]),
        ),
    ]

    for n in [3, 5, 8]:
        A = rng.normal(size=(n, n))
        A += np.eye(n) * 2.0
        b = rng.normal(size=n)
        cases.append((A, b))

    worst = 0.0
    for A, b in cases:
        ref = _oracle(A, b)
        try:
            got = sol.solve_lu(A.copy(), b.copy())
            err = _rel_err(got, ref)
            if not np.isfinite(err):
                return {"rel_err": float("inf")}
        except Exception:
            return {"rel_err": float("inf")}
        worst = max(worst, err)

    return {"rel_err": worst}
