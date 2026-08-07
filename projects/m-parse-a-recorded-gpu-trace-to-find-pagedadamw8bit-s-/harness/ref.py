import random

def generate_trace():
    random.seed(42)
    events = []
    ts = 0
    for i in range(10):
        events.append({"name": "forward", "ts": ts, "dur": 1000})
        ts += 1100
        events.append({"name": "backward", "ts": ts, "dur": 2000})
        ts += 2100

        events.append({"name": "cudaMemcpyH2D", "ts": ts - 500, "dur": 50, "args": {"bytes": 1024}})

        opt_dur = 400
        if i == 7:
            opt_dur = 15000
            events.append({"name": "cudaMemcpyH2D", "ts": ts + 100, "dur": 2000, "args": {"bytes": 8388608}})
            events.append({"name": "cudaMemcpyD2H", "ts": ts + 2500, "dur": 2000, "args": {"bytes": 8388608}})
            events.append({"name": "cudaMemcpyH2D", "ts": ts + 5000, "dur": 2000, "args": {"bytes": 8388608}})

        events.append({"name": "optimizer_step", "ts": ts, "dur": opt_dur})
        ts += opt_dur + 100
    return events

TRACE = generate_trace()

def get_optimizer_durations(events):
    return [e["dur"] for e in events if e["name"] == "optimizer_step"]

def get_spike_index(durations):
    if not durations:
        return -1
    m = max(durations)
    for i, d in enumerate(durations):
        if d == m:
            return i
    return -1

def calculate_spillover_bytes(events, step_start, step_end):
    total = 0
    for e in events:
        if e["name"] in ("cudaMemcpyH2D", "cudaMemcpyD2H"):
            if e["ts"] >= step_start and e["ts"] < step_end:
                total += e.get("args", {}).get("bytes", 0)
    return total
