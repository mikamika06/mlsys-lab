import numpy as np


def grade(sol, fx) -> dict:
    cases = [
        (
            np.array(
                [[1.0, 2.0, 3.0], [4.0, -1.0, 2.0]],
                dtype=np.float64,
            ),
            np.array(
                [[2.0, 0.0], [1.0, 3.0], [-2.0, 4.0]],
                dtype=np.float64,
            ),
            2,
        ),
        (
            np.arange(35, dtype=np.float64).reshape(5, 7) / 3.0,
            np.arange(28, dtype=np.float64).reshape(7, 4) / 5.0,
            3,
        ),
        (
            np.sin(np.arange(48, dtype=np.float64).reshape(6, 8)),
            np.cos(np.arange(24, dtype=np.float64).reshape(8, 3)),
            4,
        ),
    ]

    worst = 0.0
    for A, B, block_size in cases:
        try:
            got = sol.blocked_matmul(A, B, block_size)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = np.matmul(A, B).astype(np.float64)
        try:
            err = float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref)))
        except Exception:
            return {"max_abs_err": float("inf")}
        worst = max(worst, err)

    return {"max_abs_err": worst}
