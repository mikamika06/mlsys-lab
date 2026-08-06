def simulate_peak_memory(model_size_bytes: int, world_size: int, strategy: str) -> list[float]:
    base_overhead = 1024 * 1024 * 50
    if strategy == "FULL_SHARD":
        rank_mem = [base_overhead + (model_size_bytes / world_size) * 1.5 for _ in range(world_size)]
    elif strategy == "SHARD_GRAD_OP":
        rank_mem = [base_overhead + (model_size_bytes / world_size) + (model_size_bytes * 0.5) for _ in range(world_size)]
    else:
        rank_mem = [base_overhead + model_size_bytes for _ in range(world_size)]
    return rank_mem

def get_parameter_residency(model_size_bytes: int, strategy: str, phase: str) -> int:
    if phase == "between_forward":
        if strategy == "FULL_SHARD":
            return int(model_size_bytes // 4)
        elif strategy == "SHARD_GRAD_OP":
            return int(model_size_bytes)
    return int(model_size_bytes)

def compute_gloo_overhead(base_time: float, comm_operations: int, payload_size: int) -> float:
    coalesced_factor = 0.000001
    return float(base_time + comm_operations * (0.005 + payload_size * coalesced_factor))
