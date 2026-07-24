def grade(sol, fx) -> dict:
    import numpy as np
    cases = [
        (np.array([], dtype=np.uint64), 4096),
        (np.array([0, 1, 2], dtype=np.uint64), 1024),
        (np.array([1000, 2000, 3000, 4000], dtype=np.uint64), 1024),
        (np.arange(0, 50000, 12345, dtype=np.uint64), 4096),
        (np.random.default_rng(42).integers(0, 2**20, size=1000, dtype=np.uint64), 8192)
    ]
    ok = 1.0
    for trace, ps in cases:
        try:
            got = sol.count_distinct_pages(trace, ps)
        except Exception:
            return {"exact_match": 0.0}
        ref = np.unique(trace // ps).size
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
