def expected_tokens_per_step(accept_probs: list[float]) -> float:
    K = len(accept_probs)
    if K == 0:
        return 1.0
    total = 0.0
    prod = 1.0
    for i in range(K):
        prod *= accept_probs[i]
        total += prod
    return float(1.0 + total)
