def grade(sol, fx) -> dict:
    # Fixed set of test integers covering inside/outside the cache.
    test_values = [-10, -5, 0, 256, 257, 300, 1000]
    for n in test_values:
        try:
            got = sol.is_small_int(n)
        except Exception:
            return {"exact_match": 0.0}
        # Oracle: value lies in CPython's small‑int cache [-5,256].
        oracle = -5 <= n <= 256
        if got != oracle:
            return {"exact_match": 0.0}
    return {"exact_match": 1.0}
