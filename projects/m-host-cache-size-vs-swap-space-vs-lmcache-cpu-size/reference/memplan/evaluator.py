def calculate_overhead(host_cache, swap_space, lmcache_size):
    base_overhead = 4096
    return base_overhead + (host_cache // 2) + (swap_space // 4) + (lmcache_size // 4)


def verify_memory_bounds(allocations, limit):
    return allocations <= limit
