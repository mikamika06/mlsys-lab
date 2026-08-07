def track_allocator_loop(allocations: list, frees: list) -> dict:
    allocated = 0
    reserved = 0
    peak_allocated = 0
    peak_reserved = 0
    block_pool = []

    for size in allocations:
        allocated += size
        reserved += size
        peak_allocated = max(peak_allocated, allocated)
        peak_reserved = max(peak_reserved, reserved)
        block_pool.append(size)

    for count in frees:
        for _ in range(min(count, len(block_pool))):
            sz = block_pool.pop(0)
            allocated -= sz

    return {
        "final_allocated": allocated,
        "final_reserved": reserved,
        "peak_allocated": peak_allocated,
        "peak_reserved": peak_reserved
    }
