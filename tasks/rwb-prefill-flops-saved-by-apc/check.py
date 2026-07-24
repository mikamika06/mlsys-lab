import numpy as np
from mlsys.scorers import rel_err

def grade(sol, fx) -> dict:
    # deterministic test cases
    rng = np.random.default_rng(0)
    cases = [
        (np.array([3, 5, 2]), 1000.0),
        (np.array([]), 42.0),
        (np.arange(10), 1.5),
        (np.ones(7, dtype=int) * 4, 200.0),
        (rng.integers(0, 10, size=20), 123.456)
    ]

    max_rel = 0.0
    for counts, f in cases:
        try:
            got = sol.flops_saved_by_apc(counts, f)
        except Exception:
            return {"rel_err": float("inf")}

        # reference computed by the same algorithm
        ref = float(np.sum(counts) * f)

        # compute relative error for scalar
        rel = abs(got - ref) / (abs(ref) + 1e-12)
        if rel > max_rel:
            max_rel = rel

    return {"rel_err": max_rel}
