"""GPU utilization calculator."""

def compute_profile_utilization(kernels, trace_start_ns, trace_end_ns):
    """Compute active GPU utilization ratio merging overlapping kernel intervals."""
    if trace_end_ns <= trace_start_ns:
        return 0.0

    events_by_device = {}
    for k in kernels:
        dev = k.get("device_id", 0)
        start = max(k["start_ns"], trace_start_ns)
        end = min(k["end_ns"], trace_end_ns)
        if start < end:
            events_by_device.setdefault(dev, []).append((start, end))

    total_active_ns = 0
    num_devices = len(events_by_device) if events_by_device else 1

    for dev, intervals in events_by_device.items():
        if not intervals:
            continue
        intervals.sort(key=lambda x: x[0])
        merged = []
        cur_start, cur_end = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_end:
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))
        dev_active = sum(e - s for s, e in merged)
        total_active_ns += dev_active

    total_duration_ns = (trace_end_ns - trace_start_ns) * num_devices
    return total_active_ns / float(total_duration_ns)


def compare_gpu_utilization(report_a, report_b):
    """Compare GPU utilization between two reports and return argmin index."""
    util_a = compute_profile_utilization(
        report_a["kernels"], report_a["trace_start_ns"], report_a["trace_end_ns"]
    )
    util_b = compute_profile_utilization(
        report_b["kernels"], report_b["trace_start_ns"], report_b["trace_end_ns"]
    )
    argmin_idx = 0 if util_a <= util_b else 1
    return {
        "utilization_a": util_a,
        "utilization_b": util_b,
        "argmin_index": argmin_idx,
    }
