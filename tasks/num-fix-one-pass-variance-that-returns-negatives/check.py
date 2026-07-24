import numpy as np


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)

    cases = []
    for offset, scale, n in [
        (0.0, 3.0, 4000),
        (1e4, 2.0, 3000),
        (1e6, 4.0, 2500),
        (1e8, 2.0, 2000),
        (-1e8, 5.0, 2000),
    ]:
        x = offset + rng.normal(loc=0.0, scale=scale, size=n)
        cases.append(x)

    worst = 0.0
    for x in cases:
        ref = float(np.var(x, ddof=0))
        x_in = x.copy()
        try:
            got = sol.stable_variance(x_in)
        except Exception:
            return {"rel_err": float("inf")}
        if not np.array_equal(x_in, x):
            return {"rel_err": float("inf")}
        try:
            got = float(got)
        except Exception:
            return {"rel_err": float("inf")}
        if not np.isfinite(got) or got < 0.0:
            return {"rel_err": float("inf")}
        err = abs(got - ref) / (abs(ref) + 1e-300)
        worst = max(worst, err)

    return {"rel_err": worst}
