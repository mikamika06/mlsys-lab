def schedule_trace(reqs, slots):
    # TODO: this version delays retirement by one iteration, so completed
    # requests block newly arriving requests for an extra scheduler step.
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

        for rid in list(active):
            remaining[rid] -= 1

        finished = []
        for rid in active:
            if remaining[rid] < 0:
                finished.append(rid)

        active = [rid for rid in active if rid not in finished]

        if not waiting and not active:
            break

    return trace
