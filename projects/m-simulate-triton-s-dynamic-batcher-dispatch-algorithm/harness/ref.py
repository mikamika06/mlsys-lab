import numpy as np

def generate_arrivals(n, rate_per_sec):
    np.random.seed(42)
    inter_arrival = np.random.exponential(1e6 / rate_per_sec, n)
    return np.cumsum(inter_arrival).astype(int).tolist()

ARRIVALS = generate_arrivals(300, 200)

def dummy_compute_fn(batch_size):
    return 10000 + (batch_size * 2000)

def simulate(arrivals: list[int], max_batch_size: int, preferred: list[int], max_delay_us: int, compute_us_fn) -> list[dict]:
    time_us = 0
    queue = []
    out = []
    next_req_idx = 0
    model_ready_us = 0

    preferred_sizes = sorted([p for p in preferred if p <= max_batch_size] + [max_batch_size], reverse=True)

    while next_req_idx < len(arrivals) or queue:
        while next_req_idx < len(arrivals) and arrivals[next_req_idx] <= time_us:
            queue.append(next_req_idx)
            next_req_idx += 1

        dispatched = False
        if model_ready_us <= time_us and queue:
            max_delay_reached = (time_us >= arrivals[queue[0]] + max_delay_us)
            chosen_batch_size = 0

            if max_delay_reached:
                chosen_batch_size = min(len(queue), max_batch_size)
            else:
                for p in preferred_sizes:
                    if len(queue) >= p:
                        chosen_batch_size = p
                        break

            if chosen_batch_size > 0:
                batch_reqs = queue[:chosen_batch_size]
                queue = queue[chosen_batch_size:]
                out.append({
                    "start_us": time_us,
                    "batch_size": chosen_batch_size,
                    "request_ids": batch_reqs
                })
                model_ready_us = time_us + compute_us_fn(chosen_batch_size)
                dispatched = True

        if not dispatched:
            candidates = []
            if next_req_idx < len(arrivals):
                candidates.append(arrivals[next_req_idx])
            if model_ready_us > time_us:
                candidates.append(model_ready_us)
            if model_ready_us <= time_us and queue:
                candidates.append(arrivals[queue[0]] + max_delay_us)

            if candidates:
                time_us = max(time_us, min(candidates))
            else:
                break

    return out

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
