import numpy as np
from servebatch.boundaries import reconstruct_boundaries

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
