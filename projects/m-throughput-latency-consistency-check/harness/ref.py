def generate_valid_trace(n_events, concurrency, host_time, device_time, gap):
    events = []
    slots = [0.0] * concurrency
    for i in range(n_events):
        slot_idx = i % concurrency
        start = max(slots[slot_idx], i * gap)
        host_end = start + host_time
        device_start = host_end + 0.0001
        device_end = device_start + device_time
        slots[slot_idx] = device_end
        events.append({
            "host_start": start,
            "host_end": host_end,
            "device_start": device_start,
            "device_end": device_end
        })
    return events

def generate_inconsistent_trace():
    events = []
    for i in range(100):
        events.append({
            "host_start": i * 0.001,
            "host_end": i * 0.001 + 0.0001,
            "device_start": i * 0.001 + 0.0001,
            "device_end": i * 0.001 + 10.0
        })
    return events

def compute_metrics(events):
    if not events:
        return 0.0, 0.0, 0.0, 0.0
    t_start = min(e["host_start"] for e in events)
    t_end = max(e["device_end"] for e in events)
    t = len(events) / (t_end - t_start)
    h = sum(e["host_end"] - e["host_start"] for e in events) / len(events)
    d = sum(e["device_end"] - e["device_start"] for e in events) / len(events)
    e2e = sum(e["device_end"] - e["host_start"] for e in events) / len(events)
    return t, h, d, e2e

def consistency_error(t, e2e, c):
    if c <= 0:
        return 0.0
    return abs(t * e2e - c) / c
