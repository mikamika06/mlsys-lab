def compute_metrics(counters):
    hits = counters.get("hits", 0)
    misses = counters.get("misses", 0)
    hit_latency = counters.get("hit_latency_ms", 1.0)
    miss_latency = counters.get("miss_latency_ms", 50.0)
    bytes_per_entry = counters.get("bytes_per_entry", 1024)
    total_requests = hits + misses
    saved_latency = hits * (miss_latency - hit_latency)
    memory_cost = hits * bytes_per_entry
    return {
        "total_requests": total_requests,
        "saved_latency_ms": float(saved_latency),
        "memory_cost_bytes": int(memory_cost)
    }
