import numpy as np


def grade(sol, fx) -> dict:
    cases = [
        (5, 7, 4, 3),
        (8, 8, 8, 2),
        (11, 6, 9, 5),
        (3, 13, 4, 7),
    ]

    max_err = 0.0

    rng = np.random.default_rng(12345)
    for n, m, p, tile in cases:
        A = rng.normal(size=(n, m)).astype(np.float64)
        B = rng.normal(size=(m, p)).astype(np.float64)

        reference = np.matmul(A, B)

        try:
            candidate = sol.tiled_matmul(A, B, tile)
        except Exception:
            return {"max_abs_err": float("inf")}

        candidate = np.asarray(candidate, dtype=np.float64)
        if candidate.shape != reference.shape:
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(candidate - reference)))
        max_err = max(max_err, err)

    return {"max_abs_err": max_err}
