def measure_mps_sync_cost(host_durations, kernel_durations):
    total_host = sum(host_durations)
    cpu_curr = 0.0
    gpu_curr = 0.0
    for h_dur, k_dur in zip(host_durations, kernel_durations):
        cpu_curr += h_dur
        gpu_start = max(cpu_curr, gpu_curr)
        gpu_curr = gpu_start + k_dur
    real_wall_clock = gpu_curr
    sync_cost = max(0.0, real_wall_clock - total_host)
    return total_host, real_wall_clock, sync_cost
