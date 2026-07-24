def _ref(prev_tokens, n):
    m = len(prev_tokens)
    if m < n - 1:
        return set()
    last_gram = tuple(prev_tokens[-(n-1):]) if n > 1 else ()
    banned = set()
    for i in range(m - n + 1):
        gram = tuple(prev_tokens[i:i+n-1])
        if gram == last_gram:
            banned.add(prev_tokens[i+n-1])
    return banned


def grade(sol, fx) -> dict:
    cases = [
        ([5, 1, 3, 2, 4], 3),
        ([], 2),
        ([7, 7, 7, 7], 1),
        ([1, 2, 3, 4, 5, 6], 4),
        ([1, 2, 1, 2, 1, 2], 2),
    ]
    ok = 1.0
    for prev, n in cases:
        try:
            got = sol.no_repeat_ngram_blocking(prev, n)
            ref = _ref(prev, n)
        except Exception:
            return {"exact_match": 0.0}
        if got != ref:
            return {"exact_match": 0.0}
    return {"exact_match": ok}
