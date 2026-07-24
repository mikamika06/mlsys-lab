import numpy as np

def _ref(f0, T, t, nnz):
    ft = f0 / 2 * (1 + np.cos(np.pi * t / T))
    return round(ft * nnz)

def grade(sol, fx) -> dict:
    cases = [
        (0.1, 100, 50, 1000),
        (0.5, 200, 100, 500),
        (0.01, 50, 25, 2000),
    ]
    ok = 1.0
    for f0, T, t, nnz in cases:
        try:
            got = sol.cosine_annealed_update_fraction(f0, T, t, nnz)
        except Exception:
            ok = 0.0
            break
        ref = _ref(f0, T, t, nnz)
        if got != ref:
            ok = 0.0
            break
    return {"exact_match": ok}
