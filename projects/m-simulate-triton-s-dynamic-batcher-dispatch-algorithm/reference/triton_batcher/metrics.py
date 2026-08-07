import numpy as np

def measure_metrics(arrivals: list[int], dispatches: list[dict], compute_us_fn) -> dict:
    if not dispatches:
        return {"throughput_req_sec": 0.0, "p99_queue_delay_us": 0.0}

    delays = []
    for d in dispatches:
        for rid in d["request_ids"]:
            delays.append(d["start_us"] - arrivals[rid])

    p99 = float(np.percentile(delays, 99))
    start_t = min(arrivals)
    end_t = max(d["start_us"] + compute_us_fn(d["batch_size"]) for d in dispatches)

    dur_s = (end_t - start_t) / 1e6
    throughput = len(arrivals) / dur_s if dur_s > 0 else 0.0

    return {"throughput_req_sec": throughput, "p99_queue_delay_us": p99}
