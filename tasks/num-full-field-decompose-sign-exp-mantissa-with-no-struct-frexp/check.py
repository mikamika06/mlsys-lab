import numpy as np
from mlsys.scorers import byte_exact_fraction

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(100).astype(np.float32)*10-5  # 100 random floats
    try:
        signs, exps, mantissas = sol.decompose_floats(arr)
    except Exception:
        return {"byte_exact":0.0}
    bits = arr.view(np.uint32)
    ref_signs = (bits >> 31) & 0x1
    ref_exps = (bits >> 23) & 0xff
    ref_mantissas = bits & 0x7fffff
    ok = 1.0 if (
        byte_exact_fraction(signs, ref_signs)==1.0 and
        byte_exact_fraction(exps, ref_exps)==1.0 and
        byte_exact_fraction(mantissas, ref_mantissas)==1.0
    ) else 0.0
    return {"byte_exact":ok}
