def _oracle_propose(context, lo, hi, k):
    n = len(context)
    for L in range(hi, lo - 1, -1):
        if L <= 0 or L > n:
            continue
        suffix = context[n - L:n]
        best_i = -1
        max_i = n - 2 * L
        for i in range(0, max_i + 1):
            if context[i:i + L] == suffix:
                best_i = i  # keep scanning forward -> ends on the rightmost match
        if best_i >= 0:
            start = best_i + L
            return context[start:start + k]
    return []


def _cases(sequence):
    n = len(sequence)
    out = []
    for T in [15, 18, 33, 39, 50, 55, n - 2, n]:
        T = max(1, min(T, n))
        for lo, hi, k in [(2, 5, 3), (1, 3, 2), (3, 6, 4)]:
            out.append((sequence[:T], lo, hi, k))
    return out


def grade(sol, fx) -> dict:
    sequence = list(map(int, fx["sequence"]))

    ok = 1.0
    for context, lo, hi, k in _cases(sequence):
        expected = _oracle_propose(context, lo, hi, k)
        try:
            got = sol.propose_tokens(list(context), lo, hi, k)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
