import numpy as np

def calculate_metrics(arrivals: list[int], batches: list[dict], compute_fn) -> dict:
    if not batches:
        return {"throughput": 0.0, "p99_queue_delay": 0.0}

    start_t = min(arrivals)
    end_t = max(b["start_time"] + compute_fn(b["batch_size"]) for b in batches)

    req_queue_delays = []
    for b in batches:
        for rid in b["request_ids"]:
            req_queue_delays.append(b["start_time"] - arrivals[rid])

    p99 = float(np.percentile(req_queue_delays, 99))
    total_time_s = (end_t - start_t) / 1e6
    throughput = len(arrivals) / total_time_s if total_time_s > 0 else 0.0

    return {"throughput": throughput, "p99_queue_delay": p99}
