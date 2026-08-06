def attribute_oom(ram_total, model_bytes, kv_bytes, compute_bytes, use_mmap):
    avail = ram_total
    if not use_mmap:
        if avail < model_bytes:
            return "weights"
        avail -= model_bytes
    if avail < kv_bytes:
        return "kv"
    avail -= kv_bytes
    if avail < compute_bytes:
        return "compute"
    return "none"
