from specdec.metrics import expected_total_tokens


def evaluate_draft_throughput(
    acceptance_rates: list[float], draft_time: float, target_time: float
) -> dict[int, float]:
    """Compute expected token generation throughput for draft lengths 1..N."""
    results = {}
    for gamma in range(1, len(acceptance_rates) + 1):
        rates_sub = acceptance_rates[:gamma]
        exp_total = expected_total_tokens(rates_sub)
        total_latency = gamma * draft_time + target_time
        results[gamma] = exp_total / total_latency
    return results


def select_optimal_draft_max(
    acceptance_rates: list[float], draft_time: float, target_time: float
) -> tuple[int, float]:
    """Select the draft_max length in 1..N that maximizes expected throughput."""
    throughputs = evaluate_draft_throughput(
        acceptance_rates, draft_time, target_time
    )
    if not throughputs:
        return (1, 0.0)
    best_gamma = max(throughputs, key=lambda g: (throughputs[g], -g))
    return (best_gamma, throughputs[best_gamma])
