def discard_count(n_past, n_keep):
    n_left = n_past - n_keep
    if n_left < 0:
        n_left = 0
    return n_left // 2


def simulate(n_ctx, n_keep, n_tokens):
    resident = []
    evicted_all = []
    shift_events = []
    next_id = 0
    for _ in range(n_tokens):
        evicted = []
        if len(resident) + 1 > n_ctx:
            n_discard = discard_count(len(resident), n_keep)
            evicted = resident[n_keep:n_keep + n_discard]
            resident = resident[:n_keep] + resident[n_keep + n_discard:]
        resident.append(next_id)
        next_id += 1
        evicted_all.extend(evicted)
        shift_events.append(len(evicted))
    return {"resident": resident, "evicted": evicted_all, "shift_events": shift_events}
