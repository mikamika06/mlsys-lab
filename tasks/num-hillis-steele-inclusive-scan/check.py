import numpy as np


def _cases(rng):
    out = []
    out.append(np.array([5], dtype=np.int64))                       # N=1
    out.append(np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int64))   # N=8 (power of two)
    out.append(rng.integers(-20, 20, size=16).astype(np.int64))      # N=16, negatives
    out.append(rng.integers(-50, 50, size=13).astype(np.int64))      # N=13, not power of two
    out.append(rng.integers(0, 10, size=100).astype(np.int64))       # N=100
    out.append(np.zeros(7, dtype=np.int64))                          # all zeros
    return out


def grade(sol, fx) -> dict:
    rng = np.random.default_rng(0)
    ok = 1.0
    for x in _cases(rng):
        ref = np.cumsum(x)
        try:
            got = sol.hillis_steele_scan(x.copy())
        except Exception:
            ok = 0.0
            break
        got = np.asarray(got)
        if got.shape != ref.shape:
            ok = 0.0
            break
        if not np.issubdtype(got.dtype, np.integer):
            ok = 0.0
            break
        if not np.array_equal(got, ref):
            ok = 0.0
            break
    return {"exact_match": ok}
