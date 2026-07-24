def zig_zag_causal_assignment(num_chunks: int, num_ranks: int) -> list[int]:
    weights = [i + 1 for i in range(num_chunks)]
    order = sorted(range(num_chunks), key=lambda i: weights[i], reverse=True)

    loads = [0] * num_ranks
    direction = 1
    tie_start = 0
    result = [-1] * num_chunks

    for chunk in order:
        minimum = min(loads)
        tied = [r for r, load in enumerate(loads) if load == minimum]

        if len(tied) == 1:
            rank = tied[0]
        else:
            sequence = list(range(num_ranks))
            if direction < 0:
                sequence.reverse()
            sequence = sequence[tie_start:] + sequence[:tie_start]
            rank = next(r for r in sequence if r in tied)

        result[chunk] = rank
        loads[rank] += weights[chunk]
        direction *= -1
        tie_start = (tie_start + 1) % num_ranks

    return result
