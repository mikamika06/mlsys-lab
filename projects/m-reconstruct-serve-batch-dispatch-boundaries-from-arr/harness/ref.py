import numpy as np

CONFIGS = [
    {"arrivals": [0.0, 0.01, 0.02, 0.05, 0.12, 0.13, 0.25], "max_batch": 3, "timeout": 0.04, "concurrencies": [1, 2, 4]},
    {"arrivals": [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.035, 0.04], "max_batch": 4, "timeout": 0.02, "concurrencies": [1, 2]},
    {"arrivals": [0.1, 0.2, 0.3, 0.5, 0.51, 0.52, 0.8], "max_batch": 2, "timeout": 0.05, "concurrencies": [1, 3]}
]

def reconstruct_boundaries(arrivals, max_batch_size, timeout_s):
    if not arrivals:
        return []
    sorted_arr = sorted(arrivals)
    batches = []
    curr = [sorted_arr[0]]
    start_time = sorted_arr[0]
    for t in sorted_arr[1:]:
        if len(curr) >= max_batch_size or (t - start_time) > timeout_s:
            batches.append(curr)
            curr = [t]
            start_time = t
        else:
            curr.append(t)
    if curr:
        batches.append(curr)
    return batches

def measure_effects(arrivals, max_batch_size, timeout_s, concurrency_levels):
    batches = reconstruct_boundaries(arrivals, max_batch_size, timeout_s)
    results = {}
    for c in concurrency_levels:
        completion_times = []
        active_until = [0.0] * c
        for b in batches:
            batch_start = b[0]
            next_worker = min(range(c), key=lambda x: active_until[x])
            start_exec = max(batch_start, active_until[next_worker])
            exec_duration = 0.05 + 0.01 * len(b)
            finish_time = start_exec + exec_duration
            active_until[next_worker] = finish_time
            for _ in b:
                completion_times.append(finish_time)
        results[c] = float(np.mean(completion_times))
    return results
