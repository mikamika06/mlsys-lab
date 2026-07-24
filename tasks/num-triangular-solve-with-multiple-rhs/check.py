import numpy as np


def _oracle(L, B):
    return np.linalg.solve(L, B).astype(np.float64)


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array([[2.0]], dtype=np.float64),
            np.array([[6.0, 8.0]], dtype=np.float64),
        ),
        (
            np.array(
                [
                    [3.0, 0.0, 0.0],
                    [1.0, 4.0, 0.0],
                    [2.0, -1.0, 5.0],
                ],
                dtype=np.float64,
            ),
            np.array(
                [
                    [6.0, 3.0],
                    [7.0, 10.0],
                    [4.0, -2.0],
                ],
                dtype=np.float64,
            ),
        ),
        (
            np.tril(
                np.array(
                    [
                        [5.0, 1.0, 2.0, 3.0],
                        [4.0, 7.0, 1.0, 0.0],
                        [2.0, 6.0, 8.0, 1.0],
                        [3.0, 2.0, 4.0, 9.0],
                    ],
                    dtype=np.float64,
                )
            ),
            np.arange(12, dtype=np.float64).reshape(4, 3),
        ),
    ]

    worst = 0.0
    for L, B in cases:
        ref = _oracle(L, B)
        try:
            got = np.asarray(sol.solve_lower_multi_rhs(L, B), dtype=np.float64)
        except Exception:
            return {"rel_err": float("inf")}
        err = np.linalg.norm(got.ravel() - ref.ravel()) / (
            np.linalg.norm(ref.ravel()) + 1e-12
        )
        worst = max(worst, float(err))
    return {"rel_err": worst}
