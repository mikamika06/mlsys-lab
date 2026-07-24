def speculative_speedup(alpha: float, gamma: int, cost_ratio: float) -> float:
    """
    Expected speedup of speculative decoding over plain target-only
    autoregressive decoding.

    alpha: per-token draft-acceptance probability, 0 <= alpha <= 1.
    gamma: number of draft tokens proposed per round (positive int).
    cost_ratio: cost(one draft forward pass) / cost(one target
      verification pass) -- draft is usually much cheaper, so typically
      cost_ratio < 1.

    Return E[tokens produced per round] / (cost of one round, in units
    of a target forward pass), where a round costs `gamma * cost_ratio`
    (the gamma draft forwards) plus 1 (the single target verification
    pass that checks all gamma proposed tokens in parallel).
    """
    raise NotImplementedError('your code here')
