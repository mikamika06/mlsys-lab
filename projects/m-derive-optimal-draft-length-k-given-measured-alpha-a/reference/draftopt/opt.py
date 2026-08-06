def optimal_draft_length(alpha, cost_ratio):
    best_k = 1
    best_speedup = 0.0
    for k in range(1, 32):
        expected_accepted = sum(alpha**i for i in range(1, k + 1))
        speedup = (1.0 + expected_accepted) / (1.0 + cost_ratio * k)
        if speedup > best_speedup:
            best_speedup = speedup
            best_k = k
    return best_k
