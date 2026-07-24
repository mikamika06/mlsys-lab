def _oracle(cache_tokens, query_tokens):
    limit = min(len(cache_tokens), len(query_tokens))
    length = 0
    while length < limit and cache_tokens[length] == query_tokens[length]:
        length += 1
    return length


def grade(sol, fx) -> dict:
    cases = [
        ([10, 20, 30, 40], [10, 20, 99]),
        ([4, 7, 9, 2], [7, 9]),
        ([1, 2, 3], [1, 2, 3, 4]),
        ([5, 8, 5, 8], [8, 5, 8]),
        ([11, 12, 13], [99, 12, 13]),
        ([], [1, 2]),
        ([6, 6, 7], [6, 6, 8]),
        (list(range(20)), list(range(20))),
    ]

    ok = 1.0
    for cache, query in cases:
        expected = _oracle(cache, query)
        try:
            got = sol.prefix_reuse_length(list(cache), list(query))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
