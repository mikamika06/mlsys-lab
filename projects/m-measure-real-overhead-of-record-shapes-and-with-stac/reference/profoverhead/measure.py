def measure_option_overheads(samples):
    """Calculate average per-event overhead in nanoseconds for each profiler option."""
    totals = {}
    counts = {}
    for s in samples:
        opt = s["option"]
        base = s["baseline_ns"]
        prof = s["profiled_ns"]
        n_events = s["event_count"]
        if n_events <= 0:
            continue
        overhead = (prof - base) / float(n_events)
        totals[opt] = totals.get(opt, 0.0) + overhead
        counts[opt] = counts.get(opt, 0) + 1

    res = {}
    for opt in ["record_shapes", "with_stack", "combined"]:
        if counts.get(opt, 0) > 0:
            res[opt] = totals[opt] / counts[opt]
        else:
            res[opt] = 0.0
    return res


def calculate_doubling_ratio(baseline_event_time_ns, total_events, active_events):
    """Derive the per-event overhead ratio relative to baseline time where total execution time doubles."""
    if baseline_event_time_ns <= 0 or active_events <= 0:
        return 0.0
    ratio = float(total_events) / float(active_events)
    return ratio
