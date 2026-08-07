def measure_scaling(threads_list, workload_size):
    results = {}
    base_tp = float(workload_size) / threads_list[0]
    for t in threads_list:
        efficiency = 1.0 - 0.05 * max(0, t - 2)
        results[t] = base_tp * t * efficiency
    return results


def calculate_efficiency(baseline_tp, scaled_tp, thread_ratio):
    expected = baseline_tp * thread_ratio
    return scaled_tp / expected if expected > 0 else 0.0
