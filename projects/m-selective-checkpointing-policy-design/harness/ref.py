import numpy as np

CONFIGS = [
    {"layers": 8, "mem_costs": [10, 40, 10, 40, 10, 40, 10, 40], "compute_costs": [5, 20, 5, 20, 5, 20, 5, 20], "budget": 60},
    {"layers": 6, "mem_costs": [30, 30, 30, 30, 30, 30], "compute_costs": [10, 10, 10, 10, 10, 10], "budget": 80},
    {"layers": 4, "mem_costs": [5, 5, 5, 5], "compute_costs": [50, 50, 50, 50], "budget": 10}
]

def select_policy(config, budget):
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

def simulate_dropout(x, p, seed_fwd, seed_bwd):
    rng_fwd = np.random.default_rng(seed_fwd)
    mask_fwd = rng_fwd.binomial(1, 1 - p, size=x.shape).astype(float) / (1 - p)
    out = x * mask_fwd

    rng_bwd = np.random.default_rng(seed_bwd)
    mask_bwd = rng_bwd.binomial(1, 1 - p, size=x.shape).astype(float) / (1 - p)
    return out, mask_fwd, mask_bwd

def compile_min_cut(graph_nodes, memory_limit):
    cuts = []
    current_mem = 0
    for i, cost in enumerate(graph_nodes):
        current_mem += cost
        if current_mem > memory_limit:
            cuts.append(i)
            current_mem = cost
    return cuts
