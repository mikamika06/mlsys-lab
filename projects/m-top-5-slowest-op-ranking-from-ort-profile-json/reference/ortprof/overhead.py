def compute_overhead(events):
    node_time = sum(e.get("dur", 0) for e in events if e.get("cat") == "Node")
    prof_time = sum(e.get("dur", 0) for e in events if e.get("cat") == "Profiling")
    total_time = node_time + prof_time
    if total_time == 0:
        return 0.0
    return float(prof_time) / float(total_time)
