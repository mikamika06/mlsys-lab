def compute_realized_blocks(total_memory_bytes, reserved_bytes, utilization, block_size_bytes):
    available = total_memory_bytes * utilization - reserved_bytes
    if available <= 0:
        return 0
    return int(available // block_size_bytes)

def max_model_capacity(total_memory_bytes, reserved_bytes, utilization, block_size_bytes, dtype):
    multiplier = 0.5 if dtype == "fp8" else 1.0
    effective_block_size = block_size_bytes * multiplier
    available = total_memory_bytes * utilization - reserved_bytes
    if available <= 0:
        return 0
    return int(available // effective_block_size)

def allocate_multi_model_memory(total_memory_bytes, reserved_bytes, fractions):
    net_memory = total_memory_bytes - reserved_bytes
    if net_memory < 0:
        return [0 for _ in fractions]
    total_frac = sum(fractions)
    if total_frac <= 0:
        return [0 for _ in fractions]
    return [int(net_memory * (f / total_frac)) for f in fractions]
