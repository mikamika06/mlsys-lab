import numpy as np

def _reference(u, p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    ratio = np.where(q == 0.0, 0.0, p / q)
    ratio_clamped = np.minimum(ratio, 1.0)
    return u <= ratio_clamped

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    ok = 1.0
    tests = [
        (0.6, np.array([0.5, 1.2]), np.array([1.0, 0.8])),
        (0.2, np.array([0.3, 0.9]), np.array([0.0, 0.5])),
        (1.2, np.array([1.5, 0.4]), np.array([1.0, 0.8])),
    ]
    for _ in range(20):
        n = rng.integers(3, 15)
        p = rng.random(n) * 2.0
        q = rng.random(n) * 2.0
        u = rng.random() * 2.0
        tests.append((u, p, q))
    for u, p, q in tests:
        try:
            got = sol.classify_accept(u, p, q)
        except Exception:
            ok = 0.0
            break
        ref = _reference(u, p, q)
        if not np.array_equal(got, ref):
            ok = 0.0
            break
    return {"exact_match": ok}
