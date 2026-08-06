def measure_throughput(steps, memory_limit_enabled=False):
    base_time = steps * 0.01
    if memory_limit_enabled:
        return steps / (base_time * 1.8)
    return steps / base_time
