def recover_group_size(trace):
    if not trace:
        return 1
    max_m, max_n = max(p[0] for p in trace) + 1, max(p[1] for p in trace) + 1
    for g in range(1, max_m + 1):
        valid = True
        for pid, (m, n) in enumerate(trace):
            width = g * max_n
            group_id = pid // width
            first_pid_m = group_id * g
            g_eff = min(max_m - first_pid_m, g)
            if g_eff <= 0:
                valid = False
                break
            pid_in_group = pid % width
            expected_m = first_pid_m + (pid_in_group % g_eff)
            expected_n = pid_in_group // g_eff
            if m != expected_m or n != expected_n:
                valid = False
                break
        if valid:
            return g
    return 1
