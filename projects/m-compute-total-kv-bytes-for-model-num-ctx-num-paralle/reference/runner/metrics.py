def measure_metrics(requests, concurrency):
    total_tokens = sum(r["tokens"] for r in requests)
    base_latency = sum(r["base_latency"] for r in requests)
    contention_factor = 1.0 + max(0, concurrency - 4) * 0.05
    aggregate_latency = base_latency * contention_factor
    tok_s = total_tokens / max(aggregate_latency, 1e-6)
    per_request_latencies = [r["base_latency"] * contention_factor for r in requests]
    return {
        "aggregate_tok_s": float(tok_s),
        "per_request_latencies": per_request_latencies
    }
