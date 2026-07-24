def _oracle(existing, incoming):
    limit = min(len(existing), len(incoming))
    split = -1
    for i in range(limit):
        if existing[i] != incoming[i]:
            split = i
            break
    nodes = 2 if split != -1 else 1
    return split, nodes


def grade(sol, fx) -> dict:
    cases = [
        ([1, 2, 3, 4], [1, 2, 9, 4]),
        ([5, 8], [5, 7]),
        ([10, 20, 30], [10, 20, 30, 40]),
        ([3, 4, 5, 6], [3, 4]),
        ([9], [8]),
        ([], [1, 2]),
        ([1, 2, 3], [1, 2, 3]),
    ]

    ok = 1.0
    for existing, incoming in cases:
        expected = _oracle(list(existing), list(incoming))
        try:
            got = sol.derive_split(list(existing), list(incoming))
            got = tuple(got)
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break
    return {"exact_match": ok}
