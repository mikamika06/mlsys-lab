def bounded_slot_schedule(reqs, n_slots):
    # TODO: this admission check uses `<=` instead of `<`, so it lets one
    # extra request in even when all n_slots slots are already occupied,
    # producing n_slots + 1 active requests for that iteration.
    waiting = list(reqs)
    active = []
    remaining = {}
    trace = []

    while waiting or active:
        while len(active) <= n_slots and waiting:
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
