def compute_wait_fraction(events):
    if not events:
        return 0.0
    total_wait = sum(e["end"] - e["start"] for e in events if e.get("type") == "wait")
    start_t = min(e["start"] for e in events)
    end_t = max(e["end"] for e in events)
    span = end_t - start_t
    if span <= 0:
        return 0.0
    return float(total_wait / span)
