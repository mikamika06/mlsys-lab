import numpy as np

def _oracle(b, g):
    return np.asarray(b, dtype=np.float64) + 32.0 / np.asarray(g, dtype=np.float64)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    cases = [
        (8, 4),
        (np.array([8,16]), np.array([1,2])),
        (np.arange(1,9), np.full(8, 4)),
        # Non‑divisible group size to expose integer division bug
        (np.array([8]), np.array([3])),
        (rng.integers(4,33,size=10), rng.choice([1,2,4,8,16,32],size=10))
    ]
    max_err = 0.0
    for b,g in cases:
        try:
            got = sol.effective_bits_per_weight(b,g)
        except Exception:
            return {"max_abs_err": float("inf")}
        ref = _oracle(b,g)
        err = np.max(np.abs(got - ref))
        if err > max_err:
            max_err = err
    return {"max_abs_err": max_err}
