import numpy as np


def _oracle(A, mu):
    vals = np.linalg.eigvalsh(A)
    target = vals[np.argmin(np.abs(vals - mu))]

    x = np.ones(A.shape[0], dtype=np.float64)
    x[0] = 0.731
    x = x / np.linalg.norm(x)

    M = A - mu * np.eye(A.shape[0], dtype=np.float64)
    for _ in range(80):
        y = np.linalg.solve(M, x)
        x = y / np.linalg.norm(y)

    return float((x @ A @ x) / (x @ x))


def _rel_err(a, b):
    return float(abs(a - b) / (abs(a) + 1e-12))


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64),
            3.2,
            np.array([1.0, 0.5], dtype=np.float64),
            20,
        ),
        (
            np.array(
                [[6.0, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 2.0]],
                dtype=np.float64,
            ),
            2.1,
            np.array([0.3, 1.0, -0.4], dtype=np.float64),
            30,
        ),
        (
            np.array(
                [[9.0, -1.0, 0.5], [-1.0, 4.0, 2.0], [0.5, 2.0, 7.0]],
                dtype=np.float64,
            ),
            8.4,
            np.array([1.0, -0.2, 0.7], dtype=np.float64),
            35,
        ),
    ]

    worst = 0.0
    for A, mu, x0, iters in cases:
        try:
            got = float(sol.shifted_inverse_iteration(A, mu, x0, iters))
        except Exception:
            return {"rel_err": float("inf")}

        ref = _oracle(A, mu)
        worst = max(worst, _rel_err(ref, got))

    return {"rel_err": worst}
