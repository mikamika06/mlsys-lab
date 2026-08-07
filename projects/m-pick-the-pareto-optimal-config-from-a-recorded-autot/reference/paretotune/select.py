from paretotune.pareto import find_pareto_frontier


def select_optimal_config(records, max_registers, max_shared_mem, latency_weight=1.0, memory_weight=0.1):
    """Filter records by constraints, compute Pareto set, and return index of minimum score."""
    frontier = find_pareto_frontier(records)
    feasible = [
        r for r in frontier
        if r["num_registers"] <= max_registers and r["shared_mem_bytes"] <= max_shared_mem
    ]
    if not feasible:
        return -1

    best_idx = -1
    best_score = float("inf")
    for r in feasible:
        score = latency_weight * r["latency_us"] + memory_weight * r["memory_bytes"]
        if score < best_score:
            best_score = score
            best_idx = r["index"]

    return best_idx
