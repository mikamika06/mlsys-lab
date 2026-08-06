def run_simulation(alphas, gamma_policy):
    tokens_generated = 0
    total_cost = 0
    current_gamma = 5
    for alpha in alphas:
        if callable(gamma_policy):
            current_gamma = gamma_policy(alpha, current_gamma)
        else:
            current_gamma = int(gamma_policy)
        current_gamma = max(1, min(10, current_gamma))
        accepted = current_gamma * alpha
        tokens_generated += int(accepted) + 1
        total_cost += current_gamma + 1
    return {"tokens": tokens_generated, "cost": total_cost, "throughput": tokens_generated / max(1, total_cost)}
