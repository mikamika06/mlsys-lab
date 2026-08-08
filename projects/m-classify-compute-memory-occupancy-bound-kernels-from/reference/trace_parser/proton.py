def analyze_proton(events):
    if not events:
        return {}
    stack = []
    exclusive_time = {}
    last_time = None
    for ev in events:
        t = ev["time"]
        if last_time is not None and stack:
            top = stack[-1]
            exclusive_time[top] = exclusive_time.get(top, 0.0) + (t - last_time)
        if ev["type"] == "enter":
            stack.append(ev["region"])
        else:
            stack.pop()
        last_time = t

    total = max(e["time"] for e in events) - min(e["time"] for e in events)
    if total == 0:
        return {}
    return {k: (v / total) * 100.0 for k, v in exclusive_time.items()}
