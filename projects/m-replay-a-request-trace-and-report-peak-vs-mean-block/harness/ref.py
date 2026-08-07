import random

def get_traces():
    random.seed(42)
    traces = []
    for _ in range(5):
        num_reqs = random.randint(4, 8)
        trace = []
        for r_id in range(num_reqs):
            start = random.randint(0, 5)
            duration = random.randint(2, 6)
            blocks = random.randint(1, 10)
            trace.append({"id": r_id, "start": start, "duration": duration, "blocks": blocks})
        traces.append(trace)
    return traces

def replay_trace(trace):
    max_t = 0
    events = []
    for req in trace:
        start = req["start"]
        duration = req["duration"]
        blocks = req["blocks"]
        end = start + duration
        max_t = max(max_t, end)
        events.append((start, end, blocks))

    timeline = [0] * (max_t + 1)
    for start, end, blocks in events:
        for t in range(start, end):
            if t < len(timeline):
                timeline[t] += blocks
    return timeline

def compute_occupancy(timeline):
    if not timeline:
        return {"peak": 0, "mean": 0.0}
    peak = max(timeline)
    mean = sum(timeline) / float(len(timeline))
    return {"peak": peak, "mean": mean}
