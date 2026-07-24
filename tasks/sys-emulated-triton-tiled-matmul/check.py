import numpy as np


def grade(sol, fx) -> dict:
    cases = [
        (np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64),
         np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float64),
         1, 2, 2),
        (np.arange(35, dtype=np.float64).reshape(5, 7) / 3.0,
         np.arange(21, dtype=np.float64).reshape(7, 3) / 5.0,
         3, 2, 4),
        (np.random.default_rng(7).normal(size=(8, 9)),
         np.random.default_rng(8).normal(size=(9, 6)),
         4, 4, 3),
    ]

    worst = 0.0
    for A, B, tm, tn, tk in cases:
        try:
            got = np.asarray(sol.tiled_matmul(A, B, tm, tn, tk), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        ref = np.matmul(A, B)
        if got.shape != ref.shape:
            return {"max_abs_err": float("inf")}
        err = float(np.max(np.abs(got - ref)))
        worst = max(worst, err)

    return {"max_abs_err": worst}
