import numpy as np

def measure_communication(tokens, routing_map):
    return int(np.sum(routing_map))

def analyze_imbalance(tokens, routing_map):
    counts = np.sum(routing_map, axis=0)
    max_load = float(np.max(counts))
    mean_load = float(np.mean(counts))
    return max_load / (mean_load + 1e-5)

def group_tokens(tokens, routing_map):
    grouped = []
    for expert_idx in range(routing_map.shape[1]):
        mask = routing_map[:, expert_idx] == 1
        grouped.append(tokens[mask])
    return grouped

def overlap_computation(tokens, routing_map, compute_fn):
    res = compute_fn(tokens)
    return res

def optimized_moe_step(tokens, routing_map, compute_fn):
    grouped = group_tokens(tokens, routing_map)
    out = [compute_fn(g) if len(g) > 0 else g for g in grouped]
    return out
