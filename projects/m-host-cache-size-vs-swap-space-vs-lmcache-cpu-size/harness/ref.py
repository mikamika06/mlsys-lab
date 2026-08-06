CONFIGS = [
    {
        "host_cache_size": 512,
        "swap_space": 1024,
        "lmcache_cpu_size": 256,
        "block_size": 16
    },
    {
        "host_cache_size": 2048,
        "swap_space": 4096,
        "lmcache_cpu_size": 1024,
        "block_size": 32
    },
    {
        "host_cache_size": 128,
        "swap_space": 256,
        "lmcache_cpu_size": 64,
        "block_size": 16
    }
]


def compute_allocations(config):
    hc = config.get("host_cache_size", 0)
    sw = config.get("swap_space", 0)
    lm = config.get("lmcache_cpu_size", 0)
    block_size = config.get("block_size", 16)
    total = hc + sw + lm
    return {
        "host_cache_bytes": hc * block_size * 1024,
        "swap_bytes": sw * block_size * 1024,
        "lmcache_bytes": lm * block_size * 1024,
        "total_allocated": total * block_size * 1024
    }


def evaluate_feasibility(config, total_ram):
    allocs = compute_allocations(config)
    return allocs["total_allocated"] <= total_ram


def optimization_schedule(base_size, steps):
    schedule = []
    current = base_size
    for i in range(steps):
        current = current + (i * 128)
        schedule.append(current)
    return schedule
