import numpy as np


def grade(sol, fx) -> dict:
    rng = np.random.RandomState(7)
    shapes = [(2, 3, 2), (3, 4, 5), (5, 2, 4), (1, 3, 1), (4, 4, 4)]
    max_err = 0.0

    for (m, k, n) in shapes:
        A_np = rng.randn(m, k)
        B_np = rng.randn(k, n)
        A = A_np.tolist()
        B = B_np.tolist()

        try:
            C = sol.matmul_naive(A, B)
            C = np.asarray(C, dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        C_ref = A_np @ B_np
        if C.shape != C_ref.shape:
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(C - C_ref)))
        if err > max_err:
            max_err = err

    return {"max_abs_err": max_err}
