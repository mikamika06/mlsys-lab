def compute_overhead_ratio(baseline_time: float, logged_time: float) -> float:
    return float(logged_time) / float(baseline_time)
