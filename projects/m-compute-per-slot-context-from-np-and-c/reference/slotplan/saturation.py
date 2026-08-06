def find_saturation_point(model_bytes_base, slot_overhead_bytes, max_memory_bytes, c_total):
    """Find maximum -np before memory overhead exceeds limits."""
    if model_bytes_base > max_memory_bytes:
        return 0
    available_mem = max_memory_bytes - model_bytes_base
    per_slot_cost = slot_overhead_bytes * c_total
    if per_slot_cost <= 0:
        return 0
    max_np = available_mem // per_slot_cost
    return max(0, max_np)
