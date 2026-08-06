def extract_peak_allocated_bytes(timeline_data):
    peak = 0
    current = 0
    events = timeline_data.get("events", [])
    for ev in events:
        size = ev.get("size", 0)
        action = ev.get("action", "")
        if action == "alloc":
            current += size
            if current > peak:
                peak = current
        elif action == "free":
            current -= size
    return peak
