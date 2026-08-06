def max_concurrent_experts(cfg, memory_ceiling_bytes):
    base = int(cfg["base_size_bytes"])
    expert_size = int(cfg["expert_size_bytes"])
    if memory_ceiling_bytes < base:
        return 0
    available = memory_ceiling_bytes - base
    return min(cfg["num_experts"], max(0, available // expert_size))
