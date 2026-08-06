from scalezero.simulator import simulate_scale_to_zero

def find_optimal_timeout(traffic, cold_start_latency, max_exposure_ratio):
    total_reqs = sum(traffic)
    if total_reqs == 0:
        return 1

    for timeout in range(1, len(traffic) + 1):
        exposed, _ = simulate_scale_to_zero(traffic, timeout, cold_start_latency)
        if exposed / total_reqs <= max_exposure_ratio:
            return timeout

    return len(traffic)
