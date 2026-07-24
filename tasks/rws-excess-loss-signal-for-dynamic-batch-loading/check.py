import numpy as np
from mlsys.scorers import rel_err

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    rel_errors = []
    for size in [5, 10, 50]:
        current = rng.uniform(-1.0, 1.0, size)
        reference = rng.uniform(-1.0, 1.0, size)
        try:
            got = sol.excess_loss_signal(current, reference)
        except Exception:
            return {"rel_err": float("inf")}
        oracle = current - reference
        rel_errors.append(rel_err(oracle, got))
    max_rel_err = max(rel_errors)
    return {"rel_err": max_rel_err}
