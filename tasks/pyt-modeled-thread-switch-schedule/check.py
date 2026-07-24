def _oracle(interval, streams):
    work = [list(s) for s in streams]
    pos = [0] * len(work)
    total = 0
    current = 0
    out = []

    while any(pos[i] < len(work[i]) for i in range(len(work))):
        if pos[current] >= len(work[current]):
            current = (current + 1) % len(work)
            continue

        total += work[current][pos[current]]
        pos[current] += 1
        out.append(current)

        if total >= interval:
            total = 0
            found = False
            for step in range(1, len(work) + 1):
                candidate = (current + step) % len(work)
                if pos[candidate] < len(work[candidate]):
                    current = candidate
                    found = True
                    break
            if not found:
                break

    return out


def grade(sol, fx) -> dict:
    cases = [
        (7, [[3, 4, 5], [2, 8], [6]]),
        (5, [[5, 5], [1, 1, 1, 1], [9, 2]]),
        (10, [[1, 2, 3], [], [4, 6], [2]]),
        (3, [[2, 2, 2, 2], [1, 7], [3]]),
        (8, [[8], [8], [8]]),
    ]

    ok = 1.0
    for interval, streams in cases:
        expected = _oracle(interval, streams)
        try:
            got = sol.gil_schedule(interval, [list(s) for s in streams])
        except Exception:
            ok = 0.0
            break
        if got != expected:
            ok = 0.0
            break

    return {"exact_match": ok}
