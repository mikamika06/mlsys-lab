import numpy as np
from mlsys.scorers import rel_err

def grade(sol, fx) -> dict:
    def ref(N, d, B):
        N = int(N)
        d = int(d)
        B = int(B)
        naive = 5 * N * d * 4 + 5 * N * N * 4
        M = (N + B - 1) // B
        flash = N * d * 4 + M * (3 * N * B * 4 + 3 * B * d * 4 + N * d * 4)
        return float(naive / flash)

    cases = [
        (128, 64, 32),
        (256, 128, 64),
        (512, 256, 128),
        (1024, 512, 256),
        (10, 20, 5),
    ]

    max_err = 0.0
    for N, d, B in cases:
        try:
            got = sol.hbm_bytes_ratio(N, d, B)
            ref_val = ref(N, d, B)
        except Exception:
            return {"rel_err": float("inf")}
        if not isinstance(got, (float, int)):
            return {"rel_err": float("inf")}
        err = rel_err(np.array([ref_val]), np.array([got]))
        if err > max_err:
            max_err = err
        if err > 1e-8:
            return {"rel_err": err}
    return {"rel_err": max_err}
