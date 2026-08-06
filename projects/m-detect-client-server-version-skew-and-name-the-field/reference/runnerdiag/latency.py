def measure_first_request_cost(request_fn, iterations=5):
    """Measure duration of requests to determine cold-start overhead."""
    latencies = []
    for _ in range(iterations):
        t, res = request_fn()
        latencies.append(t)

    first_latency = latencies[0]
    warm_avg = sum(latencies[1:]) / max(1, len(latencies) - 1)
    cold_start_overhead = max(0.0, first_latency - warm_avg)

    return {
        "first_request_ms": first_latency,
        "warm_avg_ms": warm_avg,
        "cold_start_overhead_ms": cold_start_overhead,
        "all_latencies": latencies
    }
