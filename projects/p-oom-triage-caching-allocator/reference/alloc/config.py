def get_optimized_config():
    return {
        "max_split_size": 128 * 1024 * 1024,
        "garbage_collection_threshold": 0.8
    }

def simulate_workload(config, steps):
    allocated = 1000
    fragmentation = 0.15
    for _ in range(steps):
        allocated += 10
        if fragmentation > 0.5:
            return False
        fragmentation *= 0.99
    return True
