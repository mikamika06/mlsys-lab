import numpy as np

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(12345)
    shape = (5, 4)
    W = rng.standard_normal(shape).astype(np.float64)
    V = rng.standard_normal(shape).astype(np.float64) * 0.1
    s = 0.05

    try:
        got = sol.classify_rounding(W, V, s)
    except Exception:
        return {"exact_match": 0.0}

    # Reference implementation using NumPy
    r0 = np.round(W / s)
    r1 = np.round((W + V) / s)
    ref = (r1 > r0).astype(np.int8) - (r1 < r0).astype(np.int8)

    if got.shape != ref.shape:
        return {"exact_match": 0.0}
    if not np.array_equal(got, ref):
        return {"exact_match": 0.0}

    return {"exact_match": 1.0}
