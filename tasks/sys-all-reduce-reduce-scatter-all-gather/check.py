import numpy as np

def _all_reduce_ref(data, op_str):
    """Direct element-wise all-reduce via NumPy (the oracle)."""
    n = len(data)
    _op = np.sum if op_str == "sum" else np.max
    stacked = np.stack(data, axis=0)          # (n, n*k)
    reduced = _op(stacked, axis=0)             # (n*k,)
    return reduced

def grade(sol, fx) -> dict:
    cases = [
        (2, 2, "sum"),
        (2, 2, "max"),
        (3, 4, "sum"),
        (4, 3, "max"),
        (2, 8, "sum"),
        (5, 2, "sum"),
        (3, 5, "max"),
        (6, 3, "max"),
        (4, 4, "sum"),
        (3, 3, "max"),
    ]
    max_err = 0.0
    for n, k, op_str in cases:
        seed = n * 1000 + k * 100 + (7 if op_str == "max" else 3)
        rng = np.random.RandomState(seed)
        data = [rng.randn(n * k).astype(np.float64) for _ in range(n)]

        expected = _all_reduce_ref(data, op_str)   # shape (n*k,)

        try:
            scattered = sol.reduce_scatter(list(data), op_str)
            gathered = sol.all_gather(scattered)
        except Exception:
            return {"max_abs_err": float("inf")}

        if len(gathered) != n:
            return {"max_abs_err": float("inf")}

        for i in range(n):
            err = float(np.max(np.abs(gathered[i] - expected)))
            if err > max_err:
                max_err = err

    return {"max_abs_err": max_err}
