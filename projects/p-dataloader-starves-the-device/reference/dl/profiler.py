def measure_loader_fraction(step_times, wait_times):
    total_step = sum(step_times)
    total_wait = sum(wait_times)
    if total_step == 0:
        return 0.0
    return total_wait / total_step
