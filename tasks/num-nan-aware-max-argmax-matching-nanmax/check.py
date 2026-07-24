import numpy as np


def grade(sol, fx) -> dict:
    rng = np.random.RandomState(5)
    n_cases = 40
    matches = 0

    for _ in range(n_cases):
        n = int(rng.randint(5, 40))
        x = rng.randn(n)
        # scatter NaNs but always leave at least one finite value
        n_nan = int(rng.randint(0, n - 1))
        if n_nan > 0:
            nan_idx = rng.choice(n, size=n_nan, replace=False)
            x[nan_idx] = np.nan

        ref_idx = int(np.nanargmax(x))
        ref_val = float(np.nanmax(x))

        try:
            val, idx = sol.nanmax_argmax(x.copy())
            idx = int(idx)
            val = float(val)
        except Exception:
            continue

        if idx == ref_idx and abs(val - ref_val) <= 1e-9 * max(1.0, abs(ref_val)):
            matches += 1

    return {"argmax_agreement": matches / n_cases}
