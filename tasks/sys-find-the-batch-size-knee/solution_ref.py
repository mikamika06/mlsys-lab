from __future__ import annotations

def find_batch_size_knee(latencies: list[float], throughputs: list[float], slo_latency: float) -> int:
    """
    Return the index of the batch size that maximizes throughput while keeping latency <= slo_latency.
    If no such batch exists, return -1.
    """
    best_idx = -1
    max_thr = -float('inf')

    for i in range(len(latencies)):
        if latencies[i] <= slo_latency:
            current_thr = throughputs[i]
            if current_thr > max_thr:
                max_thr = current_thr
                best_idx = i

    return int(best_idx)
