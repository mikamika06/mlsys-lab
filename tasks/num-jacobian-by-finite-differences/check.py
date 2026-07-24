import numpy as np
from mlsys import scorers


def _central_diff_reference(f, x, eps):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(f(x), dtype=np.float64)
    m = y.shape[0]
    n = x.shape[0]
    J = np.empty((m, n), dtype=np.float64)

    for j in range(n):
        xp = x.copy()
        xm = x.copy()
        xp[j] += eps
        xm[j] -= eps
        yp = np.asarray(f(xp), dtype=np.float64)
        ym = np.asarray(f(xm), dtype=np.float64)
        J[:, j] = (yp - ym) / (2.0 * eps)

    return J


def grade(sol, fx) -> dict:
    cases = [
        (
            lambda x: np.array([
                x[0] ** 2 + x[1],
                np.sin(x[0]) * x[1],
            ]),
            np.array([1.3, -0.7]),
            1e-6,
        ),
        (
            lambda x: np.array([
                np.exp(x[0]) + x[1] * x[2],
                x[0] * x[1] - np.cos(x[2]),
                x[0] + x[2] ** 3,
            ]),
            np.array([0.2, -1.1, 0.8]),
            1e-6,
        ),
        (
            lambda x: np.array([
                x[0] ** 3 - 2.0 * x[1] + np.sin(x[0] * x[1]),
            ]),
            np.array([0.4, 2.5]),
            1e-6,
        ),
        (
            lambda x: np.array([
                x[0] * x[1],
                x[1] ** 2 + np.exp(x[0]),
            ]),
            np.array([-1.2, 0.6]),
            1e-6,
        ),
    ]

    errors = []
    for f, x, eps in cases:
        try:
            got = sol.jacobian_fd(f, x, eps)
            ref = _central_diff_reference(f, x, eps)
        except Exception:
            return {"rel_err": float("inf")}

        errors.append(scorers.rel_err(ref, got))

    return {"rel_err": float(max(errors))}
