def _oracle(reqs, slots):
    waiting = list(reqs)
    active = []
    remaining = {}
    trace = []

    while waiting or active:
        while len(active) < slots and waiting:
            rid, gen_len = waiting.pop(0)
            active.append(rid)
            remaining[rid] = gen_len

        if not active:
            continue

        trace.append(list(active))

        finished = []
        for rid in active:
            remaining[rid] -= 1
            if remaining[rid] == 0:
                finished.append(rid)

        active = [rid for rid in active if rid not in finished]

    return trace


def grade(sol, fx) -> dict:
    cases = [
        ([(10, 2), (20, 1), (30, 2)], 2),
        ([(1, 3), (2, 3), (3, 1), (4, 1)], 2),
        ([(5, 1), (6, 1), (7, 1), (8, 1)], 1),
        ([(9, 4), (10, 2), (11, 1), (12, 3)], 3),
        ([(100, 2), (101, 5), (102, 2), (103, 1), (104, 1)], 2),
    ]

    for reqs, slots in cases:
        try:
            got = sol.schedule_trace(list(reqs), slots)
        except Exception:
            return {"exact_match": 0.0}

        if got != _oracle(list(reqs), slots):
            return {"exact_match": 0.0}

    return {"exact_match": 1.0}
