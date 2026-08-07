import numpy as np
from tritonbatch.simulator import simulate_batcher


def optimize_preferred_batch_size(requests, throughput_floor, max_batch_size, max_queue_delay_microseconds, candidate_sizes):
    best_pref = None
    min_p99 = float("inf")

    for pref in candidate_sizes:
        if pref > max_batch_size:
            continue
        batches = simulate_batcher(requests, max_batch_size, max_queue_delay_microseconds, [pref])
        if not batches:
            continue
        total_reqs = sum(b["size"] for b in batches)
        total_time = batches[-1]["dispatch_time"] - batches[0]["requests"][0]["arrival_time"]
        throughput = total_reqs / (total_time / 1e6) if total_time > 0 else float(total_reqs)

        if throughput >= throughput_floor:
            delays = [b["max_wait"] for b in batches for _ in range(b["size"])]
            p99 = np.percentile(delays, 99) if delays else 0
            if p99 < min_p99:
                min_p99 = p99
                best_pref = pref

    if best_pref is None and candidate_sizes:
        best_pref = candidate_sizes[0]
    return best_pref
