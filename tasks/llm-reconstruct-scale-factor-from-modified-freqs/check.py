import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    rel_errs = []
    for _ in range(5):
        n = rng.integers(20, 51)
        orig = rng.uniform(0.1, 100.0, size=n).astype(np.float64)
        s_true = rng.uniform(0.3, 2.7)
        delta = rng.uniform(-1e-5, 1e-5, size=n) * s_true
        idx_outlier = rng.integers(n)
        delta[idx_outlier] = 0.001 * s_true
        mod = orig * (s_true + delta)
        try:
            got = sol.recover_scale_factor(orig.tolist(), mod.tolist())
        except Exception:
            return {"rel_err": float("inf")}
        if not isinstance(got, (float, np.floating)):
            return {"rel_err": float("inf")}
        rel_err = abs(float(got) - s_true) / abs(s_true)
        rel_errs.append(rel_err)
    max_rel_err = max(rel_errs)
    return {"rel_err": max_rel_err}
