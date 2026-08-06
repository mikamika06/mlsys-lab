def compute_recommended_wired_limit(hw_memsize_bytes: int) -> int:
    """Compute recommended set_wired_limit in bytes for a given hw.memsize."""
    if hw_memsize_bytes <= 8 * 1024 * 1024 * 1024:
        reserve_ratio = 0.25
    elif hw_memsize_bytes <= 16 * 1024 * 1024 * 1024:
        reserve_ratio = 0.20
    elif hw_memsize_bytes <= 32 * 1024 * 1024 * 1024:
        reserve_ratio = 0.15
    else:
        reserve_ratio = 0.10

    wired_bytes = int(hw_memsize_bytes * (1.0 - reserve_ratio))
    min_wired = 1024 * 1024 * 1024
    return max(min_wired, wired_bytes)


def compute_recommended_cache_limit(hw_memsize_bytes: int, active_model_bytes: int) -> int:
    """Compute recommended set_cache_limit in bytes given model size and memory limit."""
    wired = compute_recommended_wired_limit(hw_memsize_bytes)
    headroom = wired - active_model_bytes
    if headroom <= 0:
        return 64 * 1024 * 1024
    cache_bytes = int(headroom * 0.35)
    return max(64 * 1024 * 1024, cache_bytes)
