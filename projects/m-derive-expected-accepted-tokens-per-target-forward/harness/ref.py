import numpy as np


def measure_acceptance_rates(accepted_counts: list[int], draft_max: int) -> list[float]:
    if draft_max <= 0 or not accepted_counts:
        return [0.0] * max(0, draft_max)
    counts = [0] * (draft_max + 1)
    for k in accepted_counts:
        for p in range(min(max(0, k), draft_max) + 1):
            counts[p] += 1
    rates = []
    for p in range(1, draft_max + 1):
        prev = counts[p - 1]
        curr = counts[p]
        rates.append(curr / prev if prev > 0 else 0.0)
    return rates


def expected_accepted_tokens(acceptance_rates: list[float]) -> float:
    exp_tokens = 0.0
    cum_prob = 1.0
    for q in acceptance_rates:
        cum_prob *= q
        exp_tokens += cum_prob
    return exp_tokens


def expected_total_tokens(acceptance_rates: list[float]) -> float:
    return 1.0 + expected_accepted_tokens(acceptance_rates)


def evaluate_draft_throughput(
    acceptance_rates: list[float], draft_time: float, target_time: float
) -> dict[int, float]:
    results = {}
    for gamma in range(1, len(acceptance_rates) + 1):
        exp_total = expected_total_tokens(acceptance_rates[:gamma])
        total_latency = gamma * draft_time + target_time
        results[gamma] = exp_total / total_latency
    return results


def select_optimal_draft_max(
    acceptance_rates: list[float], draft_time: float, target_time: float
) -> tuple[int, float]:
    throughputs = evaluate_draft_throughput(acceptance_rates, draft_time, target_time)
    if not throughputs:
        return (1, 0.0)
    best_gamma = max(throughputs, key=lambda g: (throughputs[g], -g))
    return (best_gamma, throughputs[best_gamma])


def generate_fixtures():
    rng = np.random.default_rng(42)
    fixtures = []
    configs = [
        (8, 0.002, 0.025),
        (5, 0.005, 0.015),
        (12, 0.001, 0.040),
        (6, 0.010, 0.012),
    ]
    for draft_max, draft_time, target_time in configs:
        base_decay = float(rng.uniform(0.75, 0.95))
        probs = [base_decay ** (i + 1) for i in range(draft_max)]
        counts = []
        for _ in range(500):
            accepted = 0
            for p in probs:
                if float(rng.random()) < p:
                    accepted += 1
                else:
                    break
            counts.append(accepted)
        fixtures.append({
            "accepted_counts": counts,
            "draft_max": draft_max,
            "draft_time": draft_time,
            "target_time": target_time,
            "probs": probs,
        })
    return fixtures


TEST_TRACES = generate_fixtures()
