def compute_expected_throughput(gamma, acceptance_rate, draft_latency, target_latency):
    expected_accepted = acceptance_rate * gamma
    expected_tokens = 1.0 + expected_accepted
    iteration_latency = gamma * draft_latency + target_latency
    return expected_tokens / iteration_latency


def select_optimal_draft(candidates, target_latency, gamma):
    best_name = None
    best_throughput = -1.0
    for cand in candidates:
        tp = compute_expected_throughput(
            gamma, cand["acceptance_rate"], cand["draft_latency"], target_latency
        )
        if tp > best_throughput:
            best_throughput = tp
            best_name = cand["name"]
    return {"best_draft": best_name, "throughput": best_throughput}
