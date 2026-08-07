import numpy as np


def generate_sweep_data():
    np.random.seed(42)
    threads = [1, 2, 4, 8, 12, 16]
    base = np.array([50.0, 32.0, 22.0, 18.0, 21.0, 26.0])
    noise = np.random.normal(0, 1.0, size=(len(threads), 100))
    samples = base[:, None] + np.abs(noise)
    return threads, samples


def optimal_thread_count(threads, samples):
    p95s = [np.percentile(s, 95) for s in samples]
    best_idx = int(np.argmin(p95s))
    return threads[best_idx], best_idx


def compute_profile_gap(op_times, wall_clock):
    summed = float(np.sum(op_times))
    gap = float(wall_clock - summed)
    ratio = float(wall_clock / summed if summed > 0 else 0.0)
    return {"summed": summed, "wall_clock": wall_clock, "gap": gap, "ratio": ratio}


def max_throughput_under_sla(latencies, sla_limit):
    valid = [l for l in latencies if np.percentile(l, 95) <= sla_limit]
    if not valid:
        return 0.0
    return float(1000.0 / np.mean(valid))
