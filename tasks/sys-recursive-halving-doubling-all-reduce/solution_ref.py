def recursive_halving_doubling_all_reduce(buffers):
    p = len(buffers)
    if p == 0 or (p & (p - 1)) != 0:
        raise ValueError("rank count must be a power of two")

    work = [[float(x) for x in buf] for buf in buffers]

    # Recursive doubling all-reduce simulation. Each step exchanges the
    # accumulated buffer with the XOR partner.
    distance = 1
    while distance < p:
        updated = [[x for x in buf] for buf in work]
        for rank in range(p):
            partner = rank ^ distance
            updated[rank] = [a + b for a, b in zip(work[rank], work[partner])]
        work = updated
        distance *= 2

    return [[x for x in buf] for buf in work]
