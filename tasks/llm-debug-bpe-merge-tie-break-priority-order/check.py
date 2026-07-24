def _oracle_bpe_merge(tokens, ranks):
    pairs = []
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        if pair in ranks:
            pairs.append(pair)

    if not pairs:
        return list(tokens)

    best = min(pairs, key=lambda p: ranks[p])

    out = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def grade(sol, fx) -> dict:
    cases = [
        (
            ["l", "o", "w", "e", "r"],
            {
                ("e", "r"): 0,
                ("l", "o"): 5,
                ("o", "w"): 3,
            },
        ),
        (
            ["a", "b", "c", "a", "b"],
            {
                ("a", "b"): 4,
                ("b", "c"): 1,
            },
        ),
        (
            ["x", "y", "x", "y"],
            {
                ("x", "y"): 7,
            },
        ),
        (
            ["m", "n", "o"],
            {
                ("n", "o"): 0,
                ("m", "n"): 9,
            },
        ),
        (
            ["p", "q", "r", "s"],
            {
                ("q", "r"): 3,
                ("p", "q"): 2,
                ("r", "s"): 1,
            },
        ),
        (
            ["a", "a", "b", "a"],
            {
                ("a", "a"): 5,
                ("a", "b"): 0,
            },
        ),
    ]

    ok = 1.0
    for tokens, ranks in cases:
        expected = _oracle_bpe_merge(list(tokens), dict(ranks))
        try:
            got = sol.bpe_merge(list(tokens), dict(ranks))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
