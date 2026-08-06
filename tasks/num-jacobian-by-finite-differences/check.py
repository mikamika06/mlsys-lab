import math
import numpy as np
from mlsys import scorers


def _central_diff_reference(f, x, eps):
    x_list = [float(v) for v in x]
    y_list = [float(v) for v in f(x_list)]
    m = len(y_list)
    n = len(x_list)
    J = [[0.0 for _ in range(n)] for _ in range(m)]

    for j in range(n):
        xp = list(x_list)
        xm = list(x_list)
        xp[j] += eps
        xm[j] -= eps
        yp = [float(v) for v in f(xp)]
        ym = [float(v) for v in f(xm)]
        for i in range(m):
            J[i][j] = (yp[i] - ym[i]) / (2.0 * eps)

    return J


def grade(sol, fx) -> dict:
    cases = [
        (
            lambda x: [
                x[0] ** 2 + x[1],
                math.sin(x[0]) * x[1],
            ],
            [1.3, -0.7],
            1e-6,
        ),
        (
            lambda x: [
                math.exp(x[0]) + x[1] * x[2],
                x[0] * x[1] - math.cos(x[2]),
                x[0] + x[2] ** 3,
            ],
            [0.2, -1.1, 0.8],
            1e-6,
        ),
        (
            lambda x: [
                x[0] ** 3 - 2.0 * x[1] + math.sin(x[0] * x[1]),
            ],
            [0.4, 2.5],
            1e-6,
        ),
        (
            lambda x: [
                x[0] * x[1],
                x[1] ** 2 + math.exp(x[0]),
            ],
            [-1.2, 0.6],
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

        errors.append(scorers.rel_err(np.array(ref), np.array(got)))

    return {"rel_err": float(max(errors))}
