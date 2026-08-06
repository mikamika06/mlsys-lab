from triton_remapping.remapping import generate_grid_schedule


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
