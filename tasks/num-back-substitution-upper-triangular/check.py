import numpy as np
from mlsys import scorers

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    max_rel_err = 0.0
    for n in [5, 10, 20]:
        U = rng.standard_normal((n, n))
        U = np.triu(U)
        b = rng.standard_normal(n)
        try:
            got = sol.back_substitution(U, b)
        except Exception:
            return {"rel_err": float("inf")}
        ref = np.linalg.solve(U, b)
        rel = scorers.rel_err(ref, got)
        if rel > max_rel_err:
            max_rel_err = rel
    return {"rel_err": max_rel_err}
