def _oracle(trace, n, lookahead, pool_size):
    counts = {}
    seen = []

    def add_ngram(pos):
        if pos >= n - 1:
            ctx = tuple(trace[pos - n + 1:pos])
            tok = trace[pos]
            key = (ctx, tok)
            counts[key] = counts.get(key, 0) + 1

    for i in range(len(trace)):
        add_ngram(i)

    def ranked_pool():
        items = list(counts.items())
        items.sort(key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        return [key for key, _ in items[:pool_size]]

    verified = []
    i = n - 1
    while i < len(trace):
        current = list(trace[:i])
        proposals = []
        for _ in range(lookahead):
            if len(current) < n - 1:
                break
            ctx = tuple(current[-(n - 1):])
            choices = []
            for (c, t), freq in counts.items():
                if c == ctx:
                    choices.append((freq, t))
            if not choices:
                break
            choices.sort(key=lambda x: (-x[0], x[1]))
            nxt = choices[0][1]
            proposals.append(nxt)
            current.append(nxt)

        matched = 0
        for off, token in enumerate(proposals):
            if i + off < len(trace) and trace[i + off] == token:
                verified.append(token)
                matched += 1
            else:
                break

        for j in range(i, min(i + max(1, matched), len(trace))):
            add_ngram(j)

        i += max(1, matched)

    return verified, ranked_pool()


def grade(sol, fx) -> dict:
    cases = [
        ([1, 2, 3, 1, 2, 3, 1, 4], 3, 2, 5),
        ([5, 5, 5, 5, 6, 5, 5], 2, 3, 4),
        ([3, 1, 4, 1, 5, 9, 2], 3, 2, 8),
        ([0, 1, 0, 1, 0, 2, 0, 1], 2, 4, 3),
    ]

    ok = 1.0
    for trace, n, lookahead, pool_size in cases:
        try:
            got = sol.lookahead_pool_update_verify(
                list(trace), n, lookahead, pool_size
            )
            got = (list(got[0]), list(got[1]))
        except Exception:
            ok = 0.0
            break

        ref = _oracle(list(trace), n, lookahead, pool_size)
        if got != ref:
            ok = 0.0
            break

    return {"exact_match": ok}
