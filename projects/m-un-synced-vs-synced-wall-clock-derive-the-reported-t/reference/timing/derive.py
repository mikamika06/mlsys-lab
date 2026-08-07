def derive_reported_time_gap(launch_durations, kernel_durations):
    """Derive the gap between un-synced and synced reported times."""
    unsynced_time = sum(launch_durations)
    current_time = 0.0
    gpu_busy_end = 0.0
    for l_dur, k_dur in zip(launch_durations, kernel_durations):
        current_time += l_dur
        gpu_busy_end = max(gpu_busy_end, current_time) + k_dur
    synced_time = gpu_busy_end
    return max(0.0, synced_time - unsynced_time)
