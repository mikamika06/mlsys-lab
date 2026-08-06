def pack_experts(expert_loads, num_ranks, expert_memory_mb, rank_memory_budget_mb):
    """Packs experts onto GPUs using Longest Processing Time First greedy algorithm to minimize max rank load."""
    max_capacity = rank_memory_budget_mb // expert_memory_mb
    indexed_loads = sorted(enumerate(expert_loads), key=lambda x: x[1], reverse=True)
    rank_loads = [0] * num_ranks
    rank_counts = [0] * num_ranks
    placement = {}

    for exp_id, load in indexed_loads:
        feasible_ranks = [r for r in range(num_ranks) if rank_counts[r] < max_capacity]
        if not feasible_ranks:
            raise ValueError("Insufficient GPU memory capacity for expert placement")
        best_rank = min(feasible_ranks, key=lambda r: (rank_loads[r], rank_counts[r]))
        placement[exp_id] = best_rank
        rank_loads[best_rank] += load
        rank_counts[best_rank] += 1

    return [placement[i] for i in range(len(expert_loads))]
