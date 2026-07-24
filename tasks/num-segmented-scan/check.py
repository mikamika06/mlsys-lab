def _oracle(values, starts):
    out = []
    running = 0
    for value, start in zip(values, starts):
        if start:
            running = value
        else:
            running += value
        out.append(running)
    return out


def grade(sol, fx) -> dict:
    cases = [
        ([3, 1, 2, 5, 4, 1], [1, 0, 0, 1, 0, 0]),
        ([7], [1]),
        ([1, 2, 3, 4], [0, 0, 0, 0]),
        ([5, -2, 8, -1, 6], [1, 1, 0, 1, 0]),
        (list(range(10)), [1, 0, 0, 1, 0, 0, 1, 0, 1, 0]),
    ]

    ok = 1.0
    for values, starts in cases:
        expected = _oracle(values, starts)
        try:
            got = list(sol.segmented_scan(values, starts))
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
