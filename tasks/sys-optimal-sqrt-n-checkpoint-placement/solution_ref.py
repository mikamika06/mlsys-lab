def optimal_checkpoints(n: int) -> list[int]:
    best_k = 0
    best_cost = n
    for k in range(n):
        longest = (n + k) // (k + 1)
        cost = k + longest
        if cost < best_cost:
            best_cost = cost
            best_k = k

    k = best_k
    if k == 0:
        return []

    checkpoints = []
    previous = 0
    remaining_segments = k + 1
    for i in range(1, k + 1):
        pos = (n * i) // remaining_segments
        checkpoints.append(pos)
    return checkpoints
