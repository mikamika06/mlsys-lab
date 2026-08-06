def detect_missing_schedule(trace_events, total_steps, expected_active_steps):
    """Detect if a profiler schedule step call was missing, resulting in active profiling across all steps."""
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
