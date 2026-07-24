def _oracle(n):
    data = []
    insert_shifts = 0
    for value in range(n):
        insert_shifts += len(data)
        data.insert(0, value)

    data = []
    append_shifts = 0
    for value in range(n):
        append_shifts += 0
        data.append(value)

    return insert_shifts, append_shifts


def grade(sol, fx) -> dict:
    cases = [0, 1, 2, 5, 10, 50, 101]
    ok = 1.0
    for n in cases:
        try:
            got = tuple(sol.shift_counts(n))
        except Exception:
            ok = 0.0
            break
        if got != _oracle(n):
            ok = 0.0
            break
    return {"exact_match": ok}
