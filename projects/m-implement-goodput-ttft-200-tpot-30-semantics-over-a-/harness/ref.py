import numpy as np

def generate_fixtures():
    np.random.seed(42)
    results = []
    for i in range(20):
        arrival = float(i * 0.01)
        ttft = float(np.random.uniform(0.05, 0.25))
        first_token = arrival + ttft
        n_tokens = int(np.random.randint(5, 15))

        timestamps = [first_token]
        current = first_token
        for _ in range(n_tokens - 1):
            dt = float(np.random.uniform(0.01, 0.04))
            if i % 5 == 0 and _ == 2:
                dt += 0.2
            current += dt
            timestamps.append(current)

        results.append({
            "request_id": i,
            "status": "success" if i != 19 else "failed",
            "arrival_time": arrival,
            "first_token_time": first_token,
            "token_timestamps": timestamps,
            "preemption_time": 0.2 if i % 5 == 0 else 0.0
        })
    return results

def compute_goodput(results, max_ttft, max_tpot):
    good_count = 0
    total_count = len(results)
    if total_count == 0:
        return 0.0, 0.0, []

    accepted_requests = []
    for req in results:
        if req.get("status", "success") != "success":
            continue
        arrival = req.get("arrival_time", 0.0)
        first_token = req.get("first_token_time", arrival)
        ttft = (first_token - arrival) * 1000.0

        timestamps = req.get("token_timestamps", [])
        if len(timestamps) <= 1:
            tpot = 0.0
        else:
            intervals = np.diff(timestamps) * 1000.0
            tpot = float(np.mean(intervals))

        if ttft <= max_ttft and tpot <= max_tpot:
            good_count += 1
            accepted_requests.append(req.get("request_id"))

    goodput_ratio = good_count / total_count
    return goodput_ratio, float(good_count), accepted_requests

def analyze_preemption_gap(request_trace):
    arrival = request_trace.get("arrival_time", 0.0)
    first_token = request_trace.get("first_token_time", arrival)
    ttft = first_token - arrival
    timestamps = request_trace.get("token_timestamps", [])

    n_tokens = len(timestamps)
    e2el = (timestamps[-1] - arrival) if n_tokens > 0 else ttft

    if n_tokens <= 1:
        tpot_implied = 0.0
    else:
        tpot_implied = (timestamps[-1] - first_token) / (n_tokens - 1)

    estimated_e2el = ttft + tpot_implied * max(0, n_tokens - 1)
    preemption_overhead = request_trace.get("preemption_time", 0.0)
    actual_gap = e2el - estimated_e2el

    return {
        "e2el": float(e2el),
        "estimated_e2el": float(estimated_e2el),
        "preemption_overhead": float(preemption_overhead),
        "actual_gap": float(actual_gap)
    }

def compute_itl_and_tpot(token_timestamps):
    if len(token_timestamps) <= 1:
        return {"mean_itl": 0.0, "p99_itl": 0.0, "tpot": 0.0, "divergence": 0.0}

    timestamps = np.array(token_timestamps)
    intervals = np.diff(timestamps) * 1000.0

    mean_itl = float(np.mean(intervals))
    p99_itl = float(np.percentile(intervals, 99)) if len(intervals) > 0 else 0.0

    first_token = timestamps[0]
    last_token = timestamps[-1]
    n_gen = len(timestamps) - 1
    tpot = float((last_token - first_token) * 1000.0 / n_gen) if n_gen > 0 else 0.0

    divergence = float(p99_itl - tpot)
    return {
        "mean_itl": mean_itl,
        "p99_itl": p99_itl,
        "tpot": tpot,
        "divergence": divergence
    }
