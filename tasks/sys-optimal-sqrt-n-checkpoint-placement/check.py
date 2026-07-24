def _cost(n, checkpoints):
    if not isinstance(checkpoints, list):
        return None
    if any(not isinstance(x, int) for x in checkpoints):
        return None
    if checkpoints != sorted(checkpoints):
        return None
    if len(set(checkpoints)) != len(checkpoints):
        return None
    if any(x < 1 or x >= n for x in checkpoints):
        return None
    points = [0] + checkpoints + [n]
    segments = [b - a for a, b in zip(points, points[1:])]
    return len(checkpoints) + max(segments)


def _oracle_cost(n):
    best = None
    for k in range(n):
        # For k checkpoints there are k+1 segments. The smallest possible
        # maximum segment length is achieved by balancing the segments.
        longest = (n + k) // (k + 1)
        cost = k + longest
        if best is None or cost < best:
            best = cost
    return best


def grade(sol, fx) -> dict:
    cases = [2, 3, 8, 10, 16, 31, 64, 100]
    worst = 1.0
    for n in cases:
        try:
            got = sol.optimal_checkpoints(n)
        except Exception:
            return {"modeled_mem_access": 999.0}
        cost = _cost(n, got)
        if cost is None:
            return {"modeled_mem_access": 999.0}
        ratio = cost / _oracle_cost(n)
        if ratio > worst:
            worst = ratio
    return {"modeled_mem_access": worst}
