from batchsim.turnover import simulate_schedule


def compute_metrics(requests, max_batch_size):
    timeline = simulate_schedule(requests, max_batch_size)
    total_steps = len(timeline)
    completed_counts = [len(t["completed"]) for t in timeline]
    turnover_rate = sum(completed_counts) / total_steps if total_steps > 0 else 0.0
    completion_steps = {}
    for t in timeline:
        for cid in t["completed"]:
            completion_steps[cid] = t["step"]
    latencies = [completion_steps[r["id"]] for r in requests]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    long_reqs = [r for r in requests if r["output_len"] > 500]
    short_reqs = [r for r in requests if r["output_len"] <= 500]
    avg_short_latency = sum(completion_steps[r["id"]] for r in short_reqs) / len(short_reqs) if short_reqs else 0.0
    avg_long_latency = sum(completion_steps[r["id"]] for r in long_reqs) / len(long_reqs) if long_reqs else 0.0
    return {
        "turnover_rate": float(turnover_rate),
        "avg_latency": float(avg_latency),
        "avg_short_latency": float(avg_short_latency),
        "avg_long_latency": float(avg_long_latency),
        "total_steps": int(total_steps)
    }
