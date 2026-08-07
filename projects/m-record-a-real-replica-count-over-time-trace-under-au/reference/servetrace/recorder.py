"""Recorder for replica count over time."""

def record_trace(events, total_time):
    trace = []
    curr = 1
    event_map = dict(events)
    for t in range(total_time + 1):
        if t in event_map:
            curr = event_map[t]
        trace.append((t, curr))
    return trace
