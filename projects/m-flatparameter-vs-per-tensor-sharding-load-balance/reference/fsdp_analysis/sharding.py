def compute_load_balance(modules, world_size):
    total_params = sum(m["size"] for m in modules)
    flat_shard_size = (total_params + world_size - 1) // world_size
    rank_loads = [0] * world_size
    current_rank = 0
    current_load = 0
    for m in modules:
        size = m["size"]
        if current_load + size > flat_shard_size and current_rank < world_size - 1:
            current_rank += 1
            current_load = 0
        rank_loads[current_rank] += size
        current_load += size
    max_load = max(rank_loads)
    min_load = min(rank_loads)
    return {
        "max_load": max_load,
        "min_load": min_load,
        "imbalance_ratio": float(max_load) / float(min_load + 1)
    }
