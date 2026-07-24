import numpy as np


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = [
        rng.uniform(1e-3, 0.9, size=3000).astype(np.float64),
        rng.uniform(0.01, 0.5, size=500).astype(np.float64),
        np.full(3000, 0.5, dtype=np.float64),
        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64),
    ]

    max_rel = 0.0
    for probs in cases:
        ref = float(np.sum(np.log(probs)))
        try:
            got = sol.log_likelihood(probs)
            got = float(got)
        except Exception:
            return {"rel_err": float("inf")}
        if not np.isfinite(got):
            return {"rel_err": float("inf")}
        rel = abs(got - ref) / (abs(ref) + 1e-12)
        max_rel = max(max_rel, rel)
    return {"rel_err": max_rel}
