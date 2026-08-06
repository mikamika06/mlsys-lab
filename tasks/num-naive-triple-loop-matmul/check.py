import numpy as np
from mlsys.scorers import max_abs_err

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        (rng.integers(-5, 6, size=(3,4)), rng.integers(-5, 6, size=(4,2))),
        (rng.normal(size=(5,7)), rng.normal(size=(7,3))),
        (np.ones((1,1)), np.array([[42]])),
        (rng.random((10,10))*10, rng.random((10,10))*-3),
    ]
    max_err = 0.0
    for A,B in cases:
        A_flt = A.astype(np.float64)
        B_flt = B.astype(np.float64)
        try:
            got = sol.naive_matmul(A_flt.tolist(), B_flt.tolist())
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = np.matmul(A_flt, B_flt)
        err = max_abs_err(ref, np.array(got, dtype=np.float64))
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
