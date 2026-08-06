from triton_remapping.cache_sim import simulate_block_loads


def find_optimal_group_size(
    num_pid_m: int,
    num_pid_n: int,
    l2_cache_capacity_blocks: int,
    max_group_size: int = 32
) -> int:
    best_group_size = 1
    min_loads = float("inf")

    limit = min(num_pid_m, max_group_size)
    for g in range(1, limit + 1):
        loads = simulate_block_loads(num_pid_m, num_pid_n, g, l2_cache_capacity_blocks)
        if loads < min_loads:
            min_loads = loads
            best_group_size = g

    return best_group_size
