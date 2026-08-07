"""Thread sweep module."""

import numpy as np


def compute_p95(latencies):
    arr = np.asarray(latencies, dtype=np.float64)
    return float(np.percentile(arr, 95))


def find_optimal_thread_count(profiler, thread_counts, num_runs=100):
    best_threads = None
    min_p95 = float("inf")
    results = {}
    for t in sorted(thread_counts):
        runs = profiler.run(threads=t, num_runs=num_runs)
        p95 = compute_p95(runs)
        results[t] = p95
        if p95 < min_p95:
            min_p95 = p95
            best_threads = t
    return best_threads, results
