import math
import numpy as np


def generate_test_data(seed=123):
    rng = np.random.RandomState(seed)
    traces = []
    for _ in range(10):
        evs = []
        t = 0.0
        for _ in range(15):
            w = float(rng.uniform(0.5, 2.0))
            c = float(rng.uniform(1.0, 3.0))
            evs.append({"start": t, "end": t + w, "type": "wait"})
            t += w
            evs.append({"start": t, "end": t + c, "type": "compute"})
            t += c
        traces.append(evs)

    loads = [float(rng.uniform(1.0, 3.0)) for _ in range(10)]
    h2ds = [float(rng.uniform(0.1, 0.5)) for _ in range(10)]
    comps = [float(rng.uniform(1.5, 2.5)) for _ in range(10)]

    return {
        "traces": traces,
        "loads": loads,
        "h2ds": h2ds,
        "comps": comps
    }


def compute_wait_fraction(events):
    if not events:
        return 0.0
    total_wait = sum(e["end"] - e["start"] for e in events if e.get("type") == "wait")
    start_t = min(e["start"] for e in events)
    end_t = max(e["end"] for e in events)
    span = end_t - start_t
    if span <= 0:
        return 0.0
    return float(total_wait / span)


def simulate_double_buffer(load_times, h2d_times, compute_times):
    n = len(load_times)
    if n == 0:
        return 0.0
    t_gpu = 0.0
    t_cpu = 0.0
    for i in range(n):
        l = load_times[i]
        h = h2d_times[i]
        c = compute_times[i]
        if i == 0:
            t_cpu = l
            t_gpu = t_cpu + h + c
        else:
            t_cpu = max(t_cpu, t_cpu + l)
            t_gpu = max(t_gpu, t_cpu + h) + c
    return float(t_gpu)


def min_workers_to_saturate(load_time, consumer_time):
    if consumer_time <= 0:
        return 1
    ratio = load_time / consumer_time
    return max(1, int(math.ceil(ratio)))
