def speculative_speedup(alpha: float, gamma: int, cost_ratio: float) -> float:
    """
    Expected speedup of speculative decoding over plain target-only
    autoregressive decoding, given per-token draft-acceptance probability
    `alpha`, `gamma` draft tokens proposed per round, and `cost_ratio` =
    cost(one draft forward) / cost(one target verification pass).
    """
    expected_tokens = sum(alpha ** k for k in range(gamma + 1))
    cost_per_round = gamma * cost_ratio + 1.0
    return expected_tokens / cost_per_round
