import numpy as np

def _oracle(keys: np.ndarray, values: np.ndarray, capacity: int):
    n = keys.shape[0]
    if capacity <= 0:
        return keys[:0], values[:0]
    if capacity >= n:
        return keys.copy(), values.copy()
    norms = np.linalg.norm(keys, axis=1)
    # indices of top‑capacity by descending norm
    top_idx = np.argpartition(-norms, capacity - 1)[:capacity]
    mask = np.zeros(n, dtype=bool)
    mask[top_idx] = True
    return keys[mask], values[mask]

def grade(sol, fx) -> dict:
    ok = 1.0
    rng = np.random.default_rng(42)
    for _ in range(10):
        n = rng.integers(5, 20)
        d = rng.integers(3, 8)
        vdim = rng.integers(2, 6)
        keys = rng.standard_normal((n, d)).astype(np.float64)
        values = rng.standard_normal((n, vdim)).astype(np.float64)
        capacity = rng.integers(0, n + 5)  # allow over‑capacity
        try:
            got_keys, got_vals = sol.knorm_press(keys, values, capacity)
            ref_keys, ref_vals = _oracle(keys, values, capacity)
        except Exception:
            ok = 0.0
            break
        if not (isinstance(got_keys, np.ndarray) and isinstance(got_vals, np.ndarray)):
            ok = 0.0
            break
        if got_keys.shape != ref_keys.shape or got_vals.shape != ref_vals.shape:
            ok = 0.0
            break
        if not np.array_equal(got_keys, ref_keys):
            ok = 0.0
            break
        if not np.array_equal(got_vals, ref_vals):
            ok = 0.0
            break
    return {"exact_match": ok}
