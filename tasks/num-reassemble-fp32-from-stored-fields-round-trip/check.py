import numpy as np
from mlsys.scorers import byte_exact_fraction

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    ok = 1.0
    shapes = [(10,), (3,4), (5,2,3)]
    for shape in shapes:
        signs = rng.integers(0, 2, size=shape, dtype=np.uint8)
        exps = rng.integers(0, 256, size=shape, dtype=np.uint8)
        mantissas = rng.integers(0, 1 << 23, size=shape, dtype=np.uint32)

        bits_ref = (signs.astype(np.uint32) << 31) | \
                   (exps.astype(np.uint32) << 23) | \
                   mantissas.astype(np.uint32)
        ref = bits_ref.view(np.float32)

        try:
            out = sol.reassemble_fp32(signs, exps, mantissas)
        except Exception:
            return {"byte_exact_fraction": 0.0}

        val = byte_exact_fraction(out, ref)
        if val < 1.0 - 1e-12:
            ok = 0.0
            break

    return {"byte_exact_fraction": ok}
