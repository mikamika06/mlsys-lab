import numpy as np

def grade(sol, fx) -> dict:
    cases = [
        (128, 768, 'float32'),
        (256, 1024, 'float64'),
        (512, 4096, 'float32'),
        (1,   1,     'float32'),
        (1000,2048,'float64')
    ]
    ok = 1.0
    for S, d, dtype in cases:
        try:
            got = sol.prefill_arith_intensity(S, d, dtype)
        except Exception:
            return {"exact_match": 0}
        bsize = np.dtype(dtype).itemsize
        expected = 2 * S * d / ((2 * S + d) * bsize)
        if abs(got - expected) > 1e-12 * max(1, abs(expected)):
            ok = 0.0
            break
    return {"exact_match": ok}
