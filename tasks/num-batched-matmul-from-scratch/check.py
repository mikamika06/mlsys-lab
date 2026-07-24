import numpy as np


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ),
            np.array(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[2.0, 1.0], [1.0, 2.0]],
                ]
            ),
        ),
        (
            np.arange(24, dtype=np.float64).reshape(2, 3, 4) / 7.0,
            np.arange(40, dtype=np.float64).reshape(2, 4, 5) / 11.0,
        ),
        (
            np.array(
                [
                    [[-1.5, 2.0, 0.5]],
                    [[3.0, -4.0, 1.0]],
                    [[0.0, 5.0, -2.0]],
                ]
            ),
            np.array(
                [
                    [[2.0, -1.0], [0.5, 3.0], [4.0, 2.0]],
                    [[-2.0, 1.0], [1.0, -3.0], [0.0, 2.0]],
                    [[1.0, 4.0], [-1.0, 2.0], [3.0, -2.0]],
                ]
            ),
        ),
    ]

    worst = 0.0
    for A, B in cases:
        try:
            got = np.asarray(sol.batched_matmul(A, B), dtype=np.float64)
            ref = np.matmul(A, B)
            err = float(np.max(np.abs(got - ref)))
        except Exception:
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)

    return {"max_abs_err": worst}
