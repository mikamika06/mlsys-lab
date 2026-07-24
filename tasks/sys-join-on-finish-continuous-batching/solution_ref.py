def schedule_decode(requests, batch_size):
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

        active = [(rid, tokens) for rid, tokens in active if rid not in finished]

        while next_waiting < len(waiting) and len(active) < batch_size:
            rid, tokens = waiting[next_waiting]
            active.append((rid, tokens))
            positions[rid] = 0
            next_waiting += 1

    return output
