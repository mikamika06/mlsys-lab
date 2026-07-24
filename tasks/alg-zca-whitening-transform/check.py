import numpy as np
from mlsys.scorers import max_abs_err

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    errors = []
    for shape in [(50, 10), (100, 5), (200, 20)]:
        X = rng.standard_normal(size=shape)
        try:
            Y = sol.zca_whitening(X)
        except Exception:
            return {"max_abs_err": float("inf")}
        if Y.shape != X.shape:
            return {"max_abs_err": float("inf")}
        cov = np.cov(Y, rowvar=False)
        err = max_abs_err(cov, np.eye(cov.shape[0]))
        errors.append(err)
    return {"max_abs_err": max(errors)}
