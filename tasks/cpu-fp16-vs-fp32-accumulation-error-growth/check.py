import numpy as np
from mlsys import scorers


def _ref_accum_error_growth(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)
    s_ref = np.cumsum(x, dtype=np.float64)
    s16 = np.zeros(n, dtype=np.float64)
    s32 = np.zeros(n, dtype=np.float64)
    acc16 = np.float16(0.0)
    acc32 = np.float32(0.0)
    for i in range(n):
        acc16 = np.float16(acc16 + np.float16(x[i]))
        acc32 = np.float32(acc32 + np.float32(x[i]))
        s16[i] = float(acc16)
        s32[i] = float(acc32)
    err16 = np.abs(s16 - s_ref)
    err32 = np.abs(s32 - s_ref)
    return err16, err32


def grade(sol, fx) -> dict:
    n = 2000
    seed = 123
    ref16, ref32 = _ref_accum_error_growth(n, seed)
    try:
        got16, got32 = sol.accum_error_growth(n, seed)
        got16 = np.asarray(got16, dtype=np.float64)
        got32 = np.asarray(got32, dtype=np.float64)
    except Exception:
        return {"rel_err": 1.0}
    rel16 = scorers.rel_err(ref16, got16)
    rel32 = scorers.rel_err(ref32, got32)
    rel = max(rel16, rel32)
    return {"rel_err": rel}
