def select_checkpoint_policy(config, budget):
    mem = config["mem_costs"]
    comp = config["compute_costs"]
    n = config["layers"]
    best_score = -1.0
    best_policy = [True] * n
    total_combs = 1 << n
    for mask_int in range(total_combs):
        policy = [(mask_int >> i) & 1 == 1 for i in range(n)]
        current_mem = sum(mem[i] for i in range(n) if not policy[i])
        if current_mem <= budget:
            score = sum(comp[i] for i in range(n) if policy[i])
            if score > best_score:
                best_score = score
                best_policy = policy
    return best_policy
