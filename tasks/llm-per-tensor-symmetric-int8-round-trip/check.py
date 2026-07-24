import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    n_cases = 5
    total_rel_err = 0.0
    for _ in range(n_cases):
        shape = (rng.integers(1, 20), rng.integers(1, 10))
        W = rng.standard_normal(shape) * rng.uniform(0.1, 10)
        try:
            q, dq = sol.per_tensor_int8_round_trip(W)
        except Exception:
            return {"rel_err": float("inf")}
        # Validate shapes and dtypes
        if q.shape != W.shape or dq.shape != W.shape:
            return {"rel_err": float("inf")}
        if q.dtype.kind != 'i' or q.dtype.itemsize != 1:
            return {"rel_err": float("inf")}
        if dq.dtype != np.float64:
            return {"rel_err": float("inf")}
        # Compute relative error
        a = W.ravel()
        b = dq.ravel()
        denom = np.linalg.norm(a) + 1e-12
        rel = np.linalg.norm(b - a) / denom
        total_rel_err += rel
    avg_rel_err = total_rel_err / n_cases
    return {"rel_err": float(avg_rel_err)}
