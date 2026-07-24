import numpy as np
from mlsys.scorers import max_abs_err

def grade(sol, fx) -> dict:
    try:
        # Generate a small set of random test cases
        rng = np.random.default_rng(seed=42)
        theta = rng.uniform(1e-3, 100.0, size=10).astype(np.float64)
        factor = rng.uniform(0.1, 5.0, size=10).astype(np.float64)

        # Compute reference using NumPy
        expected = np.asarray(theta, dtype=np.float64) ** np.asarray(factor, dtype=np.float64)

        # Run candidate solution
        got = sol.scale_rope_base(theta, factor)
        if not isinstance(got, np.ndarray):
            got = np.asarray(got, dtype=np.float64)

        err = max_abs_err(expected, got)
    except Exception:
        err = float("inf")
    return {"max_abs_err": err}
