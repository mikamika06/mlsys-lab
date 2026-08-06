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
