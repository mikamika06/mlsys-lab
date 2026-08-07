def simulate_active_set(arrival_iters: list[int], gen_lens: list[int], max_num_seqs: int) -> list[list[int]]:
    """Discrete-event continuous-batching simulation: admit (FIFO, up to
    max_num_seqs) -> record the active set -> decode one token each ->
    retire anything that just reached its gen_len."""
    n = len(arrival_iters)
    waiting = sorted(range(n), key=lambda i: (int(arrival_iters[i]), i))
    active = {}  # request id -> tokens generated so far
    result = []
    t = 0
    while True:
        while waiting and len(active) < max_num_seqs and arrival_iters[waiting[0]] <= t:
            rid = waiting.pop(0)
            active[rid] = 0

        if not active and not waiting:
            break

        result.append(sorted(active.keys()))

        for rid in list(active.keys()):
            active[rid] += 1
        for rid in list(active.keys()):
            if active[rid] >= gen_lens[rid]:
                del active[rid]

        t += 1

    return result
