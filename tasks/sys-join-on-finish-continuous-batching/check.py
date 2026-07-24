def _oracle(requests, batch_size):
    waiting = [(rid, list(tokens)) for rid, tokens in requests]
    active = []
    positions = {}
    output = {rid: [] for rid, _ in requests}
    next_waiting = 0

    while next_waiting < len(waiting) and len(active) < batch_size:
        rid, tokens = waiting[next_waiting]
        active.append((rid, tokens))
        positions[rid] = 0
        next_waiting += 1

    while active:
        finished = []
        for rid, tokens in active:
            pos = positions[rid]
            output[rid].append(tokens[pos])
            positions[rid] = pos + 1
            if positions[rid] == len(tokens):
                finished.append(rid)

        if finished:
            active = [(rid, tokens) for rid, tokens in active if rid not in finished]

        while next_waiting < len(waiting) and len(active) < batch_size:
            rid, tokens = waiting[next_waiting]
            active.append((rid, tokens))
            positions[rid] = 0
            next_waiting += 1

    return output


def grade(sol, fx) -> dict:
    cases = [
        (
            [(1, [5, 6, 7]), (2, [8]), (3, [9, 10]), (4, [11])],
            2,
        ),
        (
            [(10, [1]), (11, [2, 3, 4]), (12, [5, 6]), (13, [7, 8, 9])],
            3,
        ),
        (
            [(20, [1, 2, 3, 4]), (21, [5]), (22, [6]), (23, [7, 8])],
            1,
        ),
        (
            [(30, [9, 9]), (31, [8, 8, 8]), (32, [7])],
            2,
        ),
    ]

    ok = 1.0
    for requests, batch_size in cases:
        try:
            got = sol.schedule_decode(requests, batch_size)
        except Exception:
            ok = 0.0
            break
        if got != _oracle(requests, batch_size):
            ok = 0.0
            break
    return {"exact_match": ok}
