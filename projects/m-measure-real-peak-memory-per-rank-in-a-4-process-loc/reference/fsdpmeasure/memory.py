def simulate_peak_memory(model_size_bytes: int, world_size: int, strategy: str) -> list[float]:
    base_overhead = 1024 * 1024 * 50
    if strategy == "FULL_SHARD":
        rank_mem = [base_overhead + (model_size_bytes / world_size) * 1.5 for _ in range(world_size)]
    elif strategy == "SHARD_GRAD_OP":
        rank_mem = [base_overhead + (model_size_bytes / world_size) + (model_size_bytes * 0.5) for _ in range(world_size)]
    else:
        rank_mem = [base_overhead + model_size_bytes for _ in range(world_size)]
    return rank_mem
