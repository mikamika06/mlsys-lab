def top_time_kernel(trace):
    totals = {}
    for ev in trace.get("traceEvents", []):
        name = ev.get("name", "")
        if "kernel" in name or "loss" in name or "add" in name:
            totals[name] = totals.get(name, 0) + ev.get("dur", 0)
    if not totals:
        return ""
    return max(totals, key=totals.get)
