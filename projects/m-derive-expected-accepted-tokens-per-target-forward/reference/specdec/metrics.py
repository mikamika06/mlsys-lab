def measure_acceptance_rates(accepted_counts: list[int], draft_max: int) -> list[float]:
    """Calculate conditional acceptance probabilities per draft position."""
    if draft_max <= 0:
        return []
    total_runs = len(accepted_counts)
    if total_runs == 0:
        return [0.0] * draft_max

    counts_at_least = [0] * (draft_max + 1)
    for k in accepted_counts:
        valid_k = min(max(0, k), draft_max)
        for pos in range(valid_k + 1):
            counts_at_least[pos] += 1

    rates = []
    for pos in range(1, draft_max + 1):
        prev = counts_at_least[pos - 1]
        curr = counts_at_least[pos]
        if prev == 0:
            rates.append(0.0)
        else:
            rates.append(curr / prev)
    return rates


def expected_accepted_tokens(acceptance_rates: list[float]) -> float:
    """Calculate expected accepted draft tokens per target forward pass."""
    exp_tokens = 0.0
    cum_prob = 1.0
    for q in acceptance_rates:
        cum_prob *= q
        exp_tokens += cum_prob
    return exp_tokens


def expected_total_tokens(acceptance_rates: list[float]) -> float:
    """Calculate expected total tokens generated per target forward pass."""
    return 1.0 + expected_accepted_tokens(acceptance_rates)
