import random

def generate_trace_data():
    rng = random.Random(42)
    events = []
    t = 0
    replicas = 1
    for _ in range(20):
        t += rng.randint(1, 5)
        change = rng.choice([-1, 0, 1, 2, -2])
        replicas = max(1, min(10, replicas + change))
        events.append((t, replicas))
    return events

def record_trace(events, total_time):
    trace = []
    curr = 1
    event_map = dict(events)
    for t in range(total_time + 1):
        if t in event_map:
            curr = event_map[t]
        trace.append((t, curr))
    return trace

def compare_traces(theoretical, real):
    diffs = [abs(t[1] - r[1]) for t, r in zip(theoretical, real)]
    mae = sum(diffs) / len(diffs) if diffs else 0.0
    max_diff = max(diffs) if diffs else 0
    return {"mae": mae, "max_diff": max_diff}

def detect_thrashing(trace, threshold=3):
    counts = [r[1] for r in trace]
    flips = 0
    for i in range(1, len(counts) - 1):
        if (counts[i] > counts[i-1] and counts[i] > counts[i+1]) or (counts[i] < counts[i-1] and counts[i] < counts[i+1]):
            flips += 1
    return flips >= threshold
