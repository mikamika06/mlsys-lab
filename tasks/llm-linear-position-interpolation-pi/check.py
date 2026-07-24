import numpy as np
from mlsys.scorers import max_abs_err

def _ref(positions, dim, L_train, L_target):
    positions = np.asarray(positions, dtype=np.float64)
    scale = L_target / L_train
    p_scaled = positions * scale
    freq = 1.0 / (10000 ** (np.arange(dim) / dim))
    theta = np.outer(p_scaled, freq)
    return np.sin(theta)

def grade(sol, fx) -> dict:
    cases = [
        # integer ratio
        (np.array([0, 1, 2]), 4, 10, 20),
        # non‑integer ratio to catch integer division bug
        (np.array([0, 3, 6]), 8, 7, 11),
        # larger dimension
        (np.arange(5), 16, 12, 18),
    ]
    errors = []
    for pos, dim, L_train, L_target in cases:
        try:
            cand = sol.linear_rope(pos, dim, L_train, L_target)
            ref = _ref(pos, dim, L_train, L_target)
        except Exception:
            return {"max_abs_err": float("inf")}
        errors.append(max_abs_err(ref, cand))
    max_error = max(errors)
    return {"max_abs_err": max_error}
