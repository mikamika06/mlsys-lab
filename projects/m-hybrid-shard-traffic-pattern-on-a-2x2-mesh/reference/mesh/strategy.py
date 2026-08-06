def select_strategy(model_size, memory_budget, mesh_shape=(2, 2)):
    rows, cols = mesh_shape
    ranks = rows * cols
    base = model_size // ranks
    if memory_budget >= base * 1.5:
        return "HYBRID_SHARD"
    elif memory_budget >= base:
        return "FULL_SHARD"
    else:
        return "NO_SHARD_OOM"
