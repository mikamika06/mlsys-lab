import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(5):
        n = rng.integers(2, 6)
        d = rng.integers(2, 6)
        m = rng.integers(2, 6)
        X = rng.standard_normal((n, d))
        W = rng.standard_normal((d, m))
        s = rng.uniform(0.1, 5.0, size=d)  # avoid zero
        try:
            Xp, Wp = sol.smoothing_transform(X, W, s)
        except Exception:
            return {"max_abs_err": float("inf")}
        if Xp.shape != X.shape or Wp.shape != W.shape:
            return {"max_abs_err": float("inf")}
        if Xp.dtype != np.float64 or Wp.dtype != np.float64:
            return {"max_abs_err": float("inf")}
        prod_candidate = Xp @ Wp
        prod_ref = X @ W
        err = np.max(np.abs(prod_candidate - prod_ref))
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
