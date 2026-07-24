def optimal_prefetch_distance(mem_latency: int, loop_body_cycles: int) -> int:
    return (mem_latency + loop_body_cycles - 1) // loop_body_cycles
