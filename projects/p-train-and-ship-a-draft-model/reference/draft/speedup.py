def compute_speedup(baseline_time, drafted_time):
    if drafted_time <= 0:
        return 0.0
    return float(baseline_time / drafted_time)
