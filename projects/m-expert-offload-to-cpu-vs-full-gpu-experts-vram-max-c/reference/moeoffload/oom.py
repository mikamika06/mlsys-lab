def diagnose_oom(flags: dict, available_vram: int, required_vram: int):
    issues = []
    if required_vram > available_vram:
        issues.append("insufficient_base_vram")
    if flags.get("flash_attn", False) and flags.get("cpu_offload_experts", False) and flags.get("batch_size", 1) > 32:
        issues.append("flash_attn_cpu_offload_fragmentation")
    if flags.get("tensor_parallel", 1) > 1 and not flags.get("pinned_buffers", True):
        issues.append("unpinned_tp_buffer_oom")
    return issues


def safe_allocation_limit(flags: dict, vram_total: int):
    limit = vram_total
    if flags.get("flash_attn", False):
        limit = int(limit * 0.95)
    if flags.get("cpu_offload_experts", False):
        limit = int(limit * 0.90)
    return limit
