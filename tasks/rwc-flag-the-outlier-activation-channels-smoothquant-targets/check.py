import numpy as np

def _reference_mask(X, factor):
    m = np.max(np.abs(X), axis=0)
    med = np.median(m)
    return m > (factor * med)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    ok = 1.0
    for shape in [(10,5), (50,20), (100,30), (7,3)]:
        X = rng.standard_normal(shape).astype(np.float64)
        try:
            got = sol.flag_outliers(X, factor=3.0)
        except Exception:
            ok = 0.0
            break
        ref = _reference_mask(X, 3.0)
        if not isinstance(got, np.ndarray) or got.dtype != bool:
            ok = 0.0
            break
        if not np.array_equal(got, ref):
            ok = 0.0
            break
    return {"exact_match": ok}
