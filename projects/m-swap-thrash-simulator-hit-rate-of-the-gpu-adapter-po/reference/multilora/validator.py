def validate_adapters(adapters, limits):
    max_rank = limits.get("max_rank", 64)
    max_memory = limits.get("max_memory_mb", 1024)
    total_mem = sum(a.get("memory_mb", 0) for a in adapters)
    if total_mem > max_memory:
        raise ValueError("exceeds memory limit")
    for a in adapters:
        if a.get("rank", 0) > max_rank:
            raise ValueError("exceeds rank limit")
    return {
        "max_loras": len(adapters),
        "max_lora_rank": max((a.get("rank", 0) for a in adapters), default=0),
        "lora_slots": len(adapters)
    }
