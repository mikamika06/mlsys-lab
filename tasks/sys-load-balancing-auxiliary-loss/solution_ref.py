def load_balancing_aux_loss(router_probs: list[list[float]]) -> float:
    n = len(router_probs)
    e = len(router_probs[0])
    assignments = [0] * n
    for i in range(n):
        max_val = router_probs[i][0]
        max_idx = 0
        for j in range(1, e):
            if router_probs[i][j] > max_val:
                max_val = router_probs[i][j]
                max_idx = j
        assignments[i] = max_idx

    counts = [0] * e
    for i in range(n):
        counts[assignments[i]] += 1

    f = [0.0] * e
    for j in range(e):
        f[j] = float(counts[j]) / float(n)

    p = [0.0] * e
    for j in range(e):
        col_sum = 0.0
        for i in range(n):
            col_sum += router_probs[i][j]
        p[j] = col_sum / float(n)

    total_sum = 0.0
    for j in range(e):
        total_sum += f[j] * p[j]

    return float(e * total_sum)
