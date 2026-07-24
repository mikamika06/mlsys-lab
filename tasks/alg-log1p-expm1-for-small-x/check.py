import numpy as np

def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12))

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    tests = [rng.uniform(-1e-8, 1e-8, size=10) for _ in range(5)]
    max_log_err = 0.0
    max_expm1_err = 0.0
    for x in tests:
        ref_log = np.log1p(x)
        ref_expm1 = np.expm1(x)
        try:
            got_log = sol.stable_log1p(x)
            got_expm1 = sol.stable_expm1(x)
        except Exception:
            return {"log_rel_err": 0.0, "expm1_rel_err": 0.0}
        max_log_err = max(max_log_err, _rel_err(ref_log, got_log))
        max_expm1_err = max(max_expm1_err, _rel_err(ref_expm1, got_expm1))
    return {"log_rel_err": max_log_err, "expm1_rel_err": max_expm1_err}
