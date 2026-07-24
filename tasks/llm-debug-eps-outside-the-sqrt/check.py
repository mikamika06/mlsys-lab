import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(1337)
    max_err = 0.0
    for _ in range(20):
        d = rng.integers(8, 64)
        # Small variance to make the bug visible
        x = rng.normal(0.0, 1e-3, size=d)
        gamma = rng.uniform(0.5, 2.0, size=d)
        beta = rng.normal(0.0, 0.1, size=d)
        eps = 1e-5

        # Reference: eps INSIDE the sqrt
        mu = x.mean()
        var = ((x - mu) ** 2).mean()
        std_ref = (var + eps) ** 0.5
        ref = gamma * (x - mu) / std_ref + beta

        try:
            got = np.asarray(sol.layer_norm(x, gamma, beta, eps), dtype=np.float64)
        except Exception:
            return {"max_abs_err": float("inf")}

        err = float(np.max(np.abs(got - ref)))
        if err > max_err:
            max_err = err

    return {"max_abs_err": max_err}
