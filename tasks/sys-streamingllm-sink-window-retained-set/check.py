def _oracle(n, s, w):
    retained = set()
    for i in range(n):
        if i < s or i >= max(0, n - w):
            retained.add(i)
    return sorted(retained)


def grade(sol, fx) -> dict:
    cases = [
        (0, 0, 0),
        (5, 1, 2),
        (10, 2, 4),
        (5, 4, 3),
        (8, 0, 5),
        (8, 6, 0),
        (12, 7, 7),
        (20, 3, 1),
    ]

    ok = 1.0
    for n, s, w in cases:
        expected = _oracle(n, s, w)
        try:
            got = sol.retained_kv_indices(n, s, w)
        except Exception:
            ok = 0.0
            break
        if list(got) != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
