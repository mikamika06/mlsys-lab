import numpy as np


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        rng.standard_normal((4, 4)),
        rng.standard_normal((6, 3)),   # tall
        rng.standard_normal((3, 6)),   # wide
        rng.standard_normal((5, 5)),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    ]

    worst = 0.0
    try:
        for A_arr in cases:
            A_list = A_arr.tolist()
            got = sol.reconstruct_from_svd(A_list)
            got_arr = np.asarray(got, dtype=np.float64)
            if got_arr.shape != A_arr.shape:
                return {"max_abs_err": float("inf")}
            err = float(np.max(np.abs(got_arr - A_arr)))
            worst = max(worst, err)
    except Exception:
        return {"max_abs_err": float("inf")}
    return {"max_abs_err": worst}
