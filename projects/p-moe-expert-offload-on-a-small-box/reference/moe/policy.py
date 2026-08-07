def evaluate_latency(trace, cache, budget_bytes):
    total_latency = 0.0
    for step in trace:
        for exp in step.get("activated", []):
            hit = cache.access(exp, step.get("size", 1000))
            if not hit:
                total_latency += 2.0
            else:
                total_latency += 0.1
    return total_latency
