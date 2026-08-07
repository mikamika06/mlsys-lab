def derive_timing_gap(launch_overheads, kernel_durations):
    """
    Computes (reported_unsynced_time, true_synced_time, gap) for asynchronous kernel launches.
    - launch_overheads: list of float (CPU time taken to enqueue each kernel)
    - kernel_durations: list of float (GPU execution time for each kernel)

    Unsynced time = sum(launch_overheads)
    True time = max(timeline of CPU enqueue completion vs GPU hardware completion)
    Gap = true_synced_time - reported_unsynced_time
    """
    n = len(launch_overheads)
    if n == 0:
        return 0.0, 0.0, 0.0

    unsynced_time = sum(launch_overheads)

    cpu_time = 0.0
    gpu_time = 0.0

    for i in range(n):
        cpu_time += launch_overheads[i]
        gpu_start = max(cpu_time, gpu_time)
        gpu_time = gpu_start + kernel_durations[i]

    true_synced_time = gpu_time
    gap = true_synced_time - unsynced_time
    return unsynced_time, true_synced_time, gap


def measure_mps_sync_cost(host_work_time, mps_execution_time):
    """
    Calculates total wall-clock time and synchronization overhead on MPS/GPU.
    If host dispatch finishes before GPU kernel finishes, sync waits for remaining GPU time.
    Otherwise, host was slower, so sync returns immediately with 0 extra wait.
    """
    synced_total = max(host_work_time, mps_execution_time)
    sync_wait_cost = max(0.0, mps_execution_time - host_work_time)
    return synced_total, sync_wait_cost
