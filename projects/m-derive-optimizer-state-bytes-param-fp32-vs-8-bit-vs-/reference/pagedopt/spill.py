def compute_spill_step(gpu_limit: int, base_usage_per_step: int, incremental_bytes_per_step: int) -> int:
    remaining = gpu_limit - base_usage_per_step
    if remaining <= 0:
        return 0
    return int(remaining // incremental_bytes_per_step)
