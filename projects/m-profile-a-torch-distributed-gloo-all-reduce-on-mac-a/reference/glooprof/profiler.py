def extract_gloo_self_time(trace_data):
    total = 0.0
    count = 0
    events = trace_data.get("events", [])
    for ev in events:
        if "all_reduce" in ev.get("name", ""):
            total += float(ev.get("self_time", 0.0))
            count += 1
    return total / count if count > 0 else 0.0
