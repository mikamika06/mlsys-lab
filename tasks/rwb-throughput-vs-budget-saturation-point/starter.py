def throughput_vs_budget(request_tokens, budgets):
    """Per-iteration greedy token-packing throughput curve and saturation point.

    request_tokens: list of N positive ints, in queue order.
    budgets: list of candidate per-iteration token budgets.

    For each budget B: greedily walk request_tokens in order, admitting
    request i (running_sum += request_tokens[i]) only if
    running_sum + request_tokens[i] <= B, else skip it permanently for
    this budget's pass. tokens_processed(B) = final running_sum.

    Returns (throughput_curve, saturation_budget):
      throughput_curve: [tokens_processed(b) for b in budgets].
      saturation_budget: the smallest budget at which throughput stops
        increasing with more budget (not necessarily in `budgets`).
    """
    raise NotImplementedError('your code here')
