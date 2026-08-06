import random


def generate_nsys_capture():
    rng = random.Random(42)
    kernels = []
    current_time = 0.0
    for _ in range(50):
        duration = rng.uniform(1.0, 5.0)
        gap = rng.uniform(0.1, 1.5)
        start = current_time + gap
        end = start + duration
        kernels.append({"start": start, "end": end})
        current_time = end
    wall_start = kernels[0]["start"]
    wall_end = kernels[-1]["end"]
    return kernels, wall_start, wall_end


def compute_idle_gap_ref(kernels, wall_start, wall_end):
    sorted_k = sorted(kernels, key=lambda x: x["start"])
    merged = []
    for k in sorted_k:
        if not merged:
            merged.append([k["start"], k["end"]])
        else:
            if k["start"] < merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], k["end"])
            else:
                merged.append([k["start"], k["end"]])
    active_time = sum(end - start for start, end in merged)
    wall_time = wall_end - wall_start
    idle_time = wall_time - active_time
    if idle_time < 0:
        idle_time = 0.0
    return (idle_time / wall_time) * 100.0


def generate_synthetic_kernels():
    rng = random.Random(1337)
    kernels = []
    t = 0.0
    for i in range(30):
        dur = rng.uniform(0.5, 3.0)
        gap = rng.uniform(0.2, 1.0)
        kernels.append({"start": t + gap, "end": t + gap + dur})
        t = t + gap + dur
    return kernels


def compute_warmup_ref(kernels):
    if not kernels:
        return 0.0, 0.0
    wall_start = kernels[0]["start"]
    wall_end = kernels[-1]["end"]
    idle_pct = compute_idle_gap_ref(kernels, wall_start, wall_end)
    util_pct = 100.0 - idle_pct
    return idle_pct, util_pct


def generate_sawtooth_trace():
    rng = random.Random(999)
    trace = []
    t = 0.0
    for iteration in range(10):
        for step in range(5):
            dur = rng.uniform(1.0, 2.0)
            trace.append({"start": t, "end": t + dur, "type": "kernel"})
            t += dur + 0.1
        trace.append({"start": t, "end": t + 0.5, "type": "sync"})
        t += 0.5
    return trace


def count_sync_points_ref(trace):
    count = 0
    for item in trace:
        if item.get("type") == "sync":
            count += 1
    return count
