import numpy as np
from mlsys.scorers import max_abs_err

def grade(sol, fx) -> dict:
    try:
        # Generate a small set of random test cases
        rng = np.random.default_rng(seed=42)
        theta_arr = rng.uniform(1e-3, 100.0, size=10).astype(np.float64)
        factor_arr = rng.uniform(0.1, 5.0, size=10).astype(np.float64)

        theta = theta_arr.tolist()
        factor = factor_arr.tolist()

        # Compute reference using NumPy
        expected = theta_arr ** factor_arr

        # Run candidate solution
        got = sol.scale_rope_base(theta, factor)
        if not isinstance(got, list):
            got = [got]
        got_arr = np.asarray(got, dtype=np.float64)

        err = max_abs_err(expected, got_arr)
    except Exception:
        err = float("inf")
    return {"max_abs_err": err}
