import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0
    for shape in [(5, 3), (10, 4), (7, 1), (12, 6)]:
        x = rng.standard_normal(shape).astype(np.float64)
        eps = 1e-5
        try:
            y = sol.layernorm(x, eps=eps)
        except Exception:
            return {"max_abs_err": float("inf")}
        # reference using unbiased variance (ddof=1)
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True, ddof=1)
        y_ref = (x - mean) / np.sqrt(var + eps)
        err = np.abs(y - y_ref).max()
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
