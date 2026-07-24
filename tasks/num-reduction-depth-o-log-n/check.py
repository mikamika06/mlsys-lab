def _ref_depth(N):
    """Iterative oracle: simulate ceil(N/2) halving until one element."""
    if N <= 1:
        return 0
    depth = 0
    while N > 1:
        N = (N + 1) // 2  # ceiling division by 2
        depth += 1
    return depth

def grade(sol, fx) -> dict:
    test_cases = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        15, 16, 17, 31, 32, 33, 64,
        100, 127, 128, 256,
        1000, 1024, 1025,
        2**20, 2**20 + 1,
        2**30, 2**30 + 1,
    ]
    ok = 1.0
    for n in test_cases:
        try:
            got = sol.reduction_depth(n)
        except Exception:
            ok = 0.0
            break
        if got != _ref_depth(n):
            ok = 0.0
            break
    return {"exact_match": ok}
