def _oracle(p: int, m: int) -> dict:
    start = {}
    end = {}

    for j in range(m):
        for i in range(p):
            dev_prev = end[(i, "F", j - 1)] if j > 0 else 0
            pipe_prev = end[(i - 1, "F", j)] if i > 0 else 0
            s = max(dev_prev, pipe_prev)
            start[(i, "F", j)] = s
            end[(i, "F", j)] = s + 1

    for j in reversed(range(m)):
        for i in reversed(range(p)):
            if j == m - 1:
                dev_prev = end[(i, "F", m - 1)]
            else:
                dev_prev = end[(i, "B", j + 1)]
            pipe_prev = end[(i + 1, "B", j)] if i < p - 1 else 0
            s = max(dev_prev, pipe_prev)
            start[(i, "B", j)] = s
            end[(i, "B", j)] = s + 1

    makespan = max(end.values())
    bubble_slots = p * makespan - p * 2 * m

    timeline = []
    for i in range(p):
        seq = [(start[(i, "F", j)], end[(i, "F", j)]) for j in range(m)]
        seq += [(start[(i, "B", j)], end[(i, "B", j)]) for j in reversed(range(m))]
        timeline.append(seq)

    return {"timeline": timeline, "makespan": makespan, "bubble_slots": bubble_slots}


def grade(sol, fx) -> dict:
    configs = [(2, 2), (3, 4), (4, 3), (2, 5), (5, 2), (1, 3)]
    ok = 1.0

    for p, m in configs:
        ref = _oracle(p, m)
        try:
            got = sol.gpipe_schedule(p, m)
        except Exception:
            return {"exact_match": 0.0}

        try:
            got_timeline = [[tuple(t) for t in dev] for dev in got["timeline"]]
            got_makespan = int(got["makespan"])
            got_bubble = int(got["bubble_slots"])
        except Exception:
            return {"exact_match": 0.0}

        if got_timeline != ref["timeline"]:
            ok = 0.0
        if got_makespan != ref["makespan"]:
            ok = 0.0
        if got_bubble != ref["bubble_slots"]:
            ok = 0.0

    return {"exact_match": ok}
