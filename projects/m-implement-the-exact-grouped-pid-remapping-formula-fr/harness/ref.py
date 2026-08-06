def remap_pid(pid: int, num_pid_m: int, num_pid_n: int, group_size_m: int) -> tuple[int, int]:
    num_pid_in_group = group_size_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_size_m
    group_size_m_adj = min(num_pid_m - first_pid_m, group_size_m)
    tile_id_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m_adj)
    tile_id_n = (pid % num_pid_in_group) // group_size_m_adj
    return tile_id_m, tile_id_n


def generate_grid_schedule(num_pid_m: int, num_pid_n: int, group_size_m: int) -> list[tuple[int, int]]:
    total_pids = num_pid_m * num_pid_n
    return [remap_pid(pid, num_pid_m, num_pid_n, group_size_m) for pid in range(total_pids)]


def simulate_block_loads(
    num_pid_m: int,
    num_pid_n: int,
    group_size_m: int,
    l2_cache_capacity_blocks: int
) -> int:
    schedule = generate_grid_schedule(num_pid_m, num_pid_n, group_size_m)
    l2_cache = []
    total_fetches = 0

    for tile_m, tile_n in schedule:
        block_a = ("A", tile_m)
        block_b = ("B", tile_n)

        for blk in (block_a, block_b):
            if blk in l2_cache:
                l2_cache.remove(blk)
                l2_cache.append(blk)
            else:
                total_fetches += 1
                if len(l2_cache) >= l2_cache_capacity_blocks:
                    l2_cache.pop(0)
                l2_cache.append(blk)

    return total_fetches


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


TEST_CONFIGS = [
    {"num_pid_m": 16, "num_pid_n": 16, "group_size_m": 4, "cap": 8},
    {"num_pid_m": 17, "num_pid_n": 13, "group_size_m": 5, "cap": 12},
    {"num_pid_m": 32, "num_pid_n": 64, "group_size_m": 8, "cap": 16},
    {"num_pid_m": 9, "num_pid_n": 21, "group_size_m": 4, "cap": 6},
    {"num_pid_m": 64, "num_pid_n": 32, "group_size_m": 16, "cap": 32},
]
