from specdec.metrics import expected_accepted_tokens


def find_optimal_draft_max(max_gamma, p, cost_ratio):
    best_gamma = 1
    best_throughput = -1.0
    for g in range(1, int(max_gamma) + 1):
        exp_acc = expected_accepted_tokens(g, p)
        throughput = (1.0 + exp_acc) / (1.0 + float(cost_ratio))
        if throughput > best_throughput:
            best_throughput = throughput
            best_gamma = g
    return best_gamma
