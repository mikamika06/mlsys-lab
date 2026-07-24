import numpy as np
from mlsys.scorers import rel_err

def _reference_curve(A, T):
    # Compute eigenvalues sorted by absolute value descending
    vals = np.linalg.eigvalsh(A)
    abs_vals = np.abs(vals)
    idx = np.argsort(-abs_vals)
    lam1 = vals[idx[0]]
    if len(vals) > 1:
        lam2 = vals[idx[1]]
    else:
        lam2 = 0.0
    r = abs(lam2 / lam1) if lam1 != 0 else 0.0
    return np.array([r**t for t in range(T)], dtype=np.float64)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    tests = [
        (rng.standard_normal((5, 5)), 10),
        (rng.standard_normal((8, 8)), 15),
        (rng.standard_normal((12, 12)), 20),
    ]
    errors = []
    for A_raw, T in tests:
        # Make symmetric
        A = (A_raw + A_raw.T) / 2.0
        try:
            got = sol.predict_error_curve(A, T)
        except Exception:
            return {"rel_err": 1.0}
        ref = _reference_curve(A, T)
        errors.append(rel_err(ref, got))
    avg_err = float(np.mean(errors))
    return {"rel_err": avg_err}
