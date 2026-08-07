def generate_trace(grid_m, grid_n, group_size_m):
    trace = []
    total = grid_m * grid_n
    for pid in range(total):
        width = group_size_m * grid_n
        group_id = pid // width
        first_pid_m = group_id * group_size_m
        group_size_m_eff = min(grid_m - first_pid_m, group_size_m)
        if group_size_m_eff <= 0:
            continue
        pid_in_group = pid % width
        pid_m = first_pid_m + (pid_in_group % group_size_m_eff)
        pid_n = pid_in_group // group_size_m_eff
        trace.append((pid_m, pid_n))
    return trace

TEST_CASES = [
    {"grid_m": 16, "grid_n": 8, "group_size_m": 4},
    {"grid_m": 32, "grid_n": 16, "group_size_m": 8},
    {"grid_m": 10, "grid_n": 5, "group_size_m": 4},
    {"grid_m": 64, "grid_n": 32, "group_size_m": 16},
]
