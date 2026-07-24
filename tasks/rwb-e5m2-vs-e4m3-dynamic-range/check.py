import numpy as np

def _compute_reference():
    def compute(e, m):
        bias = 2 ** (e - 1) - 1
        Emax = 2 ** e - 2
        max_norm = (2 - 2 ** (-m)) * 2 ** (Emax - bias)
        min_norm = 1.0 * 2 ** (1 - bias)
        return max_norm, min_norm

    max_e4m3, min_e4m3 = compute(4, 3)
    max_e5m2, min_e5m2 = compute(5, 2)
    return np.array([max_e4m3, min_e4m3, max_e5m2, min_e5m2], dtype=np.float64)

def grade(sol, fx) -> dict:
    try:
        cand = sol.dynamic_range()
    except Exception:
        return {"rel_err": float("inf")}

    ref = _compute_reference()

    cand_arr = np.array(cand, dtype=np.float64)
    err = np.linalg.norm(cand_arr - ref) / (np.linalg.norm(ref) + 1e-12)

    return {"rel_err": float(err)}
