import numpy as np

def _reference(arr):
    """Compute exclusive scan using NumPy."""
    # Shift right by one, inserting zero at start.
    if arr.size == 0:
        return np.empty_like(arr)
    out = np.empty_like(arr)
    out[0] = 0
    if arr.size > 1:
        out[1:] = np.cumsum(arr[:-1])
    return out

def grade(sol, fx) -> dict:
    # Generate deterministic test cases.
    rng = np.random.default_rng(42)
    cases = [
        rng.integers(-10, 10, size=0),          # empty
        rng.integers(-5, 5, size=1),            # single element
        rng.integers(-3, 3, size=5),            # small integer array
        rng.random(size=7) * 10,                # random floats
        rng.integers(-1000, 1000, size=20),     # larger integers
    ]
    ok = 1.0
    for arr in cases:
        try:
            got = sol.exclusive_scan(arr)
        except Exception:
            return {"exact_match": 0.0}
        ref = _reference(arr)
        if not np.array_equal(got, ref):
            return {"exact_match": 0.0}
    return {"exact_match": ok}
