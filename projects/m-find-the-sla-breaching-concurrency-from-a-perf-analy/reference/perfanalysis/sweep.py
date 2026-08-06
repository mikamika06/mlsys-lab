def find_max_concurrency(data, sla_ms):
    valid = [d["concurrency"] for d in data if d["p99_latency_ms"] <= sla_ms]
    return max(valid) if valid else 0
