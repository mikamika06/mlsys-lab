import numpy as np
from mlsys import scorers

def _reference(W_q, scale, zero):
    return (W_q.astype(np.float64) - zero.astype(np.float64)) * scale.astype(np.float64)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    cases = []
    for shape in [(5, 3), (10, 8), (1, 4)]:
        n, m = shape
        W_q   = rng.integers(0, 256, size=shape, dtype=np.uint8)
        scale = rng.uniform(0.01, 2.0, size=m).astype(np.float64)
        zero  = rng.integers(0, 256, size=m, dtype=np.int32)
        cases.append((W_q, scale, zero))
    errs = []
    for W_q, scale, zero in cases:
        try:
            got = sol.dequant_from_hqq_state(W_q, scale, zero)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _reference(W_q, scale, zero)
        err = scorers.max_abs_err(ref, got)
        errs.append(err)
    return {"max_abs_err": max(errs)}
