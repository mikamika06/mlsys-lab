import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(5):
        # Randomly choose 1-D or 2-D shape
        if rng.choice([True, False]):
            shape = tuple(rng.integers(1, 10, size=2))
        else:
            shape = (rng.integers(1, 10),)
        a = rng.standard_normal(shape)
        b = a + rng.standard_normal(shape) * 0.5
        try:
            got = sol.recover_sublayer_contribution(a, b)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = b - a
        err = np.max(np.abs(got - ref))
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
