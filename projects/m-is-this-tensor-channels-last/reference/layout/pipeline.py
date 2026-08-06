def steady_state_batch_time(cpu_prep_ms, transfer_ms, gpu_compute_ms, pin_memory, non_blocking):
    is_async = pin_memory and non_blocking
    if is_async:
        return max(cpu_prep_ms, transfer_ms + gpu_compute_ms)
    else:
        return max(cpu_prep_ms + transfer_ms, gpu_compute_ms)
