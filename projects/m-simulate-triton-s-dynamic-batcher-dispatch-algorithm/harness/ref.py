import numpy as np

def generate_arrivals(n=1000, rate=5000):
    np.random.seed(42)
    intervals = np.random.exponential(1e6 / rate, n)
    return np.cumsum(intervals).astype(int).tolist()

ARRIVALS = generate_arrivals(1000, 4000)

def compute_fn(b):
    return 1000 + 200 * b

def simulate(arrivals, max_batch_size, preferred_batch_sizes, max_queue_delay_us, compute_fn):
    reqs = [(i, arr) for i, arr in enumerate(arrivals)]
    reqs.sort(key=lambda x: x[1])

    Q = []
    out = []
    t = 0
    model_ready = 0

    preferred = sorted(list(set(preferred_batch_sizes + [max_batch_size])), reverse=True)

    while reqs or Q:
        while reqs and reqs[0][1] <= t:
            Q.append(reqs.pop(0))

        can_dispatch = False
        if Q and model_ready <= t:
            if t >= Q[0][1] + max_queue_delay_us:
                can_dispatch = True
            elif any(len(Q) >= p for p in preferred):
                can_dispatch = True

        if can_dispatch:
            b = 0
            if t >= Q[0][1] + max_queue_delay_us:
                b = min(len(Q), max_batch_size)
            else:
                for p in preferred:
                    if len(Q) >= p:
                        b = p
                        break

            batch = Q[:b]
            Q = Q[b:]

            out.append({
                "start_time": t,
                "batch_size": b,
                "request_ids": [req[0] for req in batch]
            })
            model_ready = t + compute_fn(b)
        else:
            next_t = float('inf')
            if reqs:
                next_t = min(next_t, float(reqs[0][1]))
            if model_ready > t:
                next_t = min(next_t, float(model_ready))
            if Q:
                next_t = min(next_t, float(Q[0][1] + max_queue_delay_us))

            if next_t == float('inf'):
                break
            t = int(next_t)

    return out

def calculate_metrics(arrivals, batches, compute_fn):
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
