import numpy as np

def _ref(sizes, max_split_size_mb, min_remainder_mb=1.0):
    sizes = np.asarray(sizes)
    return (sizes <= max_split_size_mb) | ((sizes - max_split_size_mb) >= min_remainder_mb)

def grade(sol, fx) -> dict:
    rng = np.random.default_rng(42)
    ok = 1.0
    for _ in range(5):
        n = rng.integers(1, 20)
        sizes = rng.uniform(0.1, 10.0, size=n).astype(np.float64)
        max_split_size_mb = rng.uniform(0.5, 5.0)
        min_remainder_mb = 1.0
        try:
            got = sol.classify_blocks(sizes, max_split_size_mb, min_remainder_mb)
            expected = _ref(sizes, max_split_size_mb, min_remainder_mb)
        except Exception:
            return {"exact_match": 0.0}
        if not np.array_equal(got, expected):
            return {"exact_match": 0.0}
    return {"exact_match": ok}
