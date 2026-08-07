def compute_overlap_percentage(trace):
    comp_intervals = []
    comm_intervals = []
    for ev in trace.get("traceEvents", []):
        ts = ev.get("ts", 0)
        dur = ev.get("dur", 0)
        cat = ev.get("cat")
        if cat == "compute":
            comp_intervals.append((ts, ts + dur))
        elif cat == "comm":
            comm_intervals.append((ts, ts + dur))

    total_comm = sum(e - s for s, e in comm_intervals)
    if total_comm == 0:
        return 0.0

    overlap_time = 0.0
    for cs, ce in comm_intervals:
        intersect = 0.0
        for ms, me in comp_intervals:
            start = max(cs, ms)
            end = min(ce, me)
            if start < end:
                intersect += (end - start)
        overlap_time += intersect

    return float(overlap_time / total_comm * 100.0)
