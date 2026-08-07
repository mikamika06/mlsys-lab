def compute_cache_bytes(total_memory, free_memory, fraction, mode="free"):
    base = free_memory if mode == "free" else total_memory
    return int(base * fraction)
