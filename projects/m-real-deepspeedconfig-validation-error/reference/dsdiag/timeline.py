def compute_overlap_ratio(events):
    total_time = max(e["end"] for e in events) - min(e["start"] for e in events)
    overlap = 0
    for c in events:
        if c["name"] != "compute":
            continue
        for o in events:
            if o["name"] != "comm":
                continue
            latest_start = max(c["start"], o["start"])
            earliest_end = min(c["end"], o["end"])
            if latest_start < earliest_end:
                overlap += (earliest_end - latest_start)
    if total_time == 0:
        return 0.0
    return float(overlap) / float(total_time)
