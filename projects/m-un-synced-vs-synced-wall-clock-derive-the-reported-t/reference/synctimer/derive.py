def derive_reported_gap(launch_times, execution_times):
    if not launch_times:
        return 0.0, 0.0, 0.0
    unsynced_time = sum(launch_times)
    cpu_curr = 0.0
    gpu_curr = 0.0
    for l_time, e_time in zip(launch_times, execution_times):
        cpu_curr += l_time
        gpu_start = max(cpu_curr, gpu_curr)
        gpu_curr = gpu_start + e_time
    synced_time = gpu_curr
    gap = synced_time - unsynced_time
    return unsynced_time, synced_time, gap
