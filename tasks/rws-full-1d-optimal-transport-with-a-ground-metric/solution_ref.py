def ot_cost_1d(positions: list[float], p: list[float], q: list[float]) -> float:
    """1-D optimal transport cost with ground cost |x - y|, via the
    closed-form cumulative-CDF-difference formula (exact for convex
    1-D ground costs -- no LP needed).
    """
    n = len(positions)
    order = sorted(range(n), key=lambda i: positions[i])

    x = [0.0] * n
    p_sorted = [0.0] * n
    q_sorted = [0.0] * n

    for i in range(n):
        idx = order[i]
        x[i] = positions[idx]
        p_sorted[i] = p[idx]
        q_sorted[i] = q[idx]

    P = [0.0] * n
    Q = [0.0] * n
    curr_p = 0.0
    curr_q = 0.0
    for i in range(n):
        curr_p += p_sorted[i]
        curr_q += q_sorted[i]
        P[i] = curr_p
        Q[i] = curr_q

    total = 0.0
    for i in range(n - 1):
        gap = x[i + 1] - x[i]
        diff = P[i] - Q[i]
        total += abs(diff) * gap

    return float(total)
