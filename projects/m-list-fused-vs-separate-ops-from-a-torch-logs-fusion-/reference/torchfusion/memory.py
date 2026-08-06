def compute_memory(fused_groups, inplace=True):
    total_bytes = 0
    buffer_sizes = {}
    for i, group in enumerate(fused_groups):
        size = len(group) * 1024
        buffer_sizes[i] = size
    if inplace:
        total_bytes = max(buffer_sizes.values()) if buffer_sizes else 0
    else:
        total_bytes = sum(buffer_sizes.values())
    return total_bytes
