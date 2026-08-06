def max_model_capacity(total_memory_bytes, reserved_bytes, utilization, block_size_bytes, dtype):
    multiplier = 0.5 if dtype == "fp8" else 1.0
    effective_block_size = block_size_bytes * multiplier
    available = total_memory_bytes * utilization - reserved_bytes
    if available <= 0:
        return 0
    return int(available // effective_block_size)
