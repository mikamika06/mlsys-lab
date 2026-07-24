import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    errors = []
    for _ in range(10):
        n = rng.integers(0, 10000)
        arr = rng.standard_normal(n).astype(np.float64)
        try:
            got = sol.pairwise_sum(arr)
        except Exception:
            return {"rel_err": float("inf")}
        ref = np.sum(arr, dtype=np.float64)
        if not isinstance(got, (float, np.floating)):
            return {"rel_err": float("inf")}
        err = abs(float(got) - float(ref)) / (abs(ref)+1e-12)
        errors.append(err)
    rel_err = max(errors) if errors else 0.0
    return {"rel_err": rel_err}
