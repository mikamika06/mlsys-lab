def _oracle(trace, chunk_size):
    """
    Compute the total number of tokens saved by prefix‑chunk reuse.
    This is a pure Python implementation that enumerates all contiguous
    subsequences up to `chunk_size` and stores them in a set for O(1) lookup.
    """
    cache = set()
    total_saved = 0
    for req in trace:
        # longest cached prefix
        max_len = 0
        n = len(req)
        limit = min(n, chunk_size)
        for l in range(1, limit + 1):
            if tuple(req[:l]) in cache and l > max_len:
                max_len = l
        total_saved += max_len

        # add all contiguous subsequences up to chunk_size
        for i in range(n):
            for l in range(1, min(chunk_size, n - i) + 1):
                cache.add(tuple(req[i:i + l]))
    return total_saved


def grade(sol, fx) -> dict:
    """
    The grader runs a handful of deterministic test cases and compares
    the candidate's result to the oracle computed above.
    """
    cases = [
        # (trace, chunk_size)
        ([[1, 2, 3], [2, 3, 4], [1, 2, 5]], 512),
        ([[10, 20, 30, 40], [20, 30, 50], [10, 20, 30, 60]], 512),
        ([], 512),
        ([list(range(100))], 512),
        ([ [i for i in range(j)] for j in range(1,6) ], 3),
    ]

    ok = 1.0
    for trace, size in cases:
        try:
            got = sol.prefix_reuse_savings(trace, size)
        except Exception:
            return {"exact_match": 0.0}
        expected = _oracle(trace, size)
        if got != expected:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
