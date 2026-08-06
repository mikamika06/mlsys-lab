def compute_overhead_reduction(before_trace, after_trace):
    b_total = sum(item.get("launch_delay", 0) + item.get("driver_wait", 0) for item in before_trace)
    a_total = sum(item.get("launch_delay", 0) + item.get("driver_wait", 0) for item in after_trace)
    if b_total == 0:
        return 1.0
    return float(b_total - a_total) / float(b_total)
