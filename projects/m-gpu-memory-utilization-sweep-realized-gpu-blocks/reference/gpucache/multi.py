def allocate_multi_model_memory(total_memory_bytes, reserved_bytes, fractions):
    net_memory = total_memory_bytes - reserved_bytes
    if net_memory < 0:
        return [0 for _ in fractions]
    total_frac = sum(fractions)
    if total_frac <= 0:
        return [0 for _ in fractions]
    return [int(net_memory * (f / total_frac)) for f in fractions]
