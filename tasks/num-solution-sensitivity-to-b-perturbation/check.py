import numpy as np


def _reference(A, b, delta):
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)

    x = np.linalg.solve(A, b)
    xp = np.linalg.solve(A, b + delta)
    return float(np.linalg.norm(xp - x) / (np.linalg.norm(x) + 1e-12))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[3.0, 1.0], [1.0, 2.0]]),
            np.array([4.0, 5.0]),
            np.array([0.01, -0.02]),
        ),
        (
            np.array([[1.0, 0.99], [0.99, 0.98]]),
            np.array([2.0, 1.0]),
            np.array([0.001, -0.001]),
        ),
        (
            np.array([[10.0, 2.0, 1.0], [2.0, 8.0, 3.0], [1.0, 3.0, 9.0]]),
            np.array([1.0, -2.0, 3.0]),
            np.array([0.1, 0.0, -0.05]),
        ),
        (
            np.array([[2.0, 0.5], [0.5, 1.0]]),
            np.array([-1.0, 2.0]),
            np.array([0.001, 0.003]),
        ),
    ]

    try:
        ref = np.array([_reference(*case) for case in cases], dtype=np.float64)
        got = np.array(
            [float(sol.solution_sensitivity(*case)) for case in cases],
            dtype=np.float64,
        )
    except Exception:
        return {"rel_err": 1.0}

    err = float(np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12))
    return {"rel_err": err}
