import numpy as np
from mlsys.scorers import max_abs_err

def grade(sol, fx) -> dict:
    def ref_unpack(packed, scale):
        lo = packed & 0xF
        hi = (packed >> 4) & 0xF
        vals = np.concatenate([lo, hi], axis=0).astype(np.int16)
        vals = np.where(vals > 7, vals - 16, vals)
        return scale * vals.astype(np.float32)

    errors = []
    for _ in range(5):
        n_bytes = np.random.randint(10, 200)
        packed_vals = np.random.randint(-8, 8, size=2*n_bytes, dtype=np.int16)
        unsigned = np.where(packed_vals < 0, packed_vals + 16, packed_vals).astype(np.uint8)
        packed = (unsigned[1::2] << 4) | unsigned[0::2]
        scale = np.random.uniform(0.01, 5.0)
        try:
            got = sol.unpack_dequant_qint4(packed, float(scale))
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = ref_unpack(packed, scale)
        errors.append(max_abs_err(ref, got))
    return {"max_abs_err": max(errors)}
