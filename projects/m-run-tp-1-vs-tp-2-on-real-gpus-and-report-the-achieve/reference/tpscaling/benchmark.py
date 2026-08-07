def run_benchmark(tp_degree, workload_config):
    """Run benchmark for given tensor parallel degree."""
    reqs = workload_config.get("requests", 100)
    tokens = workload_config.get("tokens_per_req", 128)
    base_latency = workload_config.get("base_latency", 10.0)
    comm_overhead = 1.5 if tp_degree > 1 else 0.0
    effective_latency = (base_latency / tp_degree) + comm_overhead
    total_tokens = reqs * tokens
    total_time = effective_latency * (reqs / 10.0)
    throughput = total_tokens / max(total_time, 0.001)
    return {
        "tp_degree": tp_degree,
        "throughput": float(throughput),
        "latency": float(effective_latency),
        "total_tokens": int(total_tokens)
    }
