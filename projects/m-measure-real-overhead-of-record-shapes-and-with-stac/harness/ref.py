BENCHMARK_SAMPLES = [
    {"option": "record_shapes", "baseline_ns": 5000000, "profiled_ns": 6250000, "event_count": 5000},
    {"option": "record_shapes", "baseline_ns": 10000000, "profiled_ns": 12800000, "event_count": 10000},
    {"option": "with_stack", "baseline_ns": 5000000, "profiled_ns": 8500000, "event_count": 5000},
    {"option": "with_stack", "baseline_ns": 10000000, "profiled_ns": 17200000, "event_count": 10000},
    {"option": "combined", "baseline_ns": 5000000, "profiled_ns": 9800000, "event_count": 5000},
    {"option": "combined", "baseline_ns": 10000000, "profiled_ns": 19600000, "event_count": 10000},
]


def measure_option_overheads(samples):
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
    if baseline_event_time_ns <= 0 or active_events <= 0:
        return 0.0
    ratio = float(total_events) / float(active_events)
    return ratio


def detect_missing_schedule(trace_events, total_steps, expected_active_steps):
    step_counts = {}
    for evt in trace_events:
        st = evt.get("step")
        if st is not None:
            step_counts[st] = step_counts.get(st, 0) + 1

    total_profiled_events = len(trace_events)
    active_step_count = len(step_counts)

    missing = False
    if active_step_count > expected_active_steps:
        missing = True
    elif active_step_count == 0 and total_steps > 0 and total_profiled_events > 0:
        missing = True

    return {
        "missing_schedule": missing,
        "active_steps": active_step_count,
        "total_steps": total_steps,
        "total_profiled_events": total_profiled_events,
    }
