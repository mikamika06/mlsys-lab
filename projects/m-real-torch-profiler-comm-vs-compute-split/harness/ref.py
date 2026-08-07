import random


def generate_trace(seed, enable_overlap=True):
    rng = random.Random(seed)
    events = []
    current_ts = 1000
    for i in range(20):
        comp_dur = rng.randint(50, 200)
        events.append({
            "name": f"aten::matmul_{i}",
            "cat": "compute",
            "ts": current_ts,
            "dur": comp_dur
        })
        comm_dur = rng.randint(20, 80)
        if enable_overlap:
            comm_ts = current_ts + rng.randint(0, 10)
        else:
            comm_ts = current_ts + comp_dur + 5
        events.append({
            "name": "nccl:all_reduce",
            "cat": "comm",
            "ts": comm_ts,
            "dur": comm_dur
        })
        current_ts = max(current_ts + comp_dur, comm_ts + comm_dur) + 10
    return {"traceEvents": events}


def parse_split(trace):
    comp_time = 0.0
    comm_time = 0.0
    for ev in trace.get("traceEvents", []):
        cat = ev.get("cat")
        dur = ev.get("dur", 0.0)
        if cat == "compute":
            comp_time += dur
        elif cat == "comm":
            comm_time += dur
    return {"compute_time": comp_time, "comm_time": comm_time}


def compute_overlap(trace):
    comp_intervals = []
    comm_intervals = []
    for ev in trace.get("traceEvents", []):
        ts = ev.get("ts", 0)
        dur = ev.get("dur", 0)
        cat = ev.get("cat")
        if cat == "compute":
            comp_intervals.append((ts, ts + dur))
        elif cat == "comm":
            comm_intervals.append((ts, ts + dur))

    total_comm = sum(e - s for s, e in comm_intervals)
    if total_comm == 0:
        return 0.0

    overlap_time = 0.0
    for cs, ce in comm_intervals:
        intersect = 0.0
        for ms, me in comp_intervals:
            start = max(cs, ms)
            end = min(ce, me)
            if start < end:
                intersect += (end - start)
        overlap_time += intersect

    return float(overlap_time / total_comm * 100.0)


def classify_trace(trace):
    pct = compute_overlap(trace)
    return "enabled" if pct > 25.0 else "disabled"


TEST_TRACES = [
    generate_trace(42, enable_overlap=True),
    generate_trace(123, enable_overlap=False),
    generate_trace(999, enable_overlap=True)
]
