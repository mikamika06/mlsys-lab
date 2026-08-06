def compute_realized_blocks(total_memory_bytes, reserved_bytes, utilization, block_size_bytes):
    available = total_memory_bytes * utilization - reserved_bytes
    if available <= 0:
        return 0
    return int(available // block_size_bytes)
