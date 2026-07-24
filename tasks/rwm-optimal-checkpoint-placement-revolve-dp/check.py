def _oracle(L, C):
    if L <= 1:
        return 0
    if C <= 0:
        return L * (L - 1) // 2
    if C >= L:
        return 0
    best = None
    for k in range(1, L):
        value = k + _oracle(k, C) + _oracle(L - k, C - 1)
        if best is None or value < best:
            best = value
    return best


def grade(sol, fx) -> dict:
    cases = []
    for L in range(0, 16):
        for C in range(0, 16):
            cases.append((L, C))

    ok = 1.0
    for L, C in cases:
        try:
            got = sol.optimal_recompute(L, C)
        except Exception:
            ok = 0.0
            break
        if not isinstance(got, int) or got != _oracle(L, C):
            ok = 0.0
            break
    return {"exact_match": ok}
