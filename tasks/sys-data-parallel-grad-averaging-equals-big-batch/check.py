from mlsys.scorers import max_abs_err
import numpy as np

def _full_gradient(X, y):
    n = X.shape[0]
    return 2.0 / n * X.T @ y

def grade(sol, fx) -> dict:
    ok = True
    max_err = 0.0
    for seed in [1, 2, 3]:
        rng = np.random.default_rng(seed)
        n = rng.integers(20, 100)
        d = rng.integers(5, 15)
        num_shards = rng.choice([2, 4, 8])
        if n % num_shards != 0:
            n = (n // num_shards) * num_shards
        X = rng.standard_normal((n, d))
        y = rng.standard_normal(n)
        try:
            user_grad = sol.data_parallel_grad_avg(X, y, num_shards)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref_grad = _full_gradient(X, y)
        err = max_abs_err(ref_grad, user_grad)
        if err > max_err:
            max_err = err
        if err > 1e-6:
            ok = False
    return {"max_abs_err": max_err}
