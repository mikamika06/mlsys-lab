def sweep_parameters(sizes, timeouts, workload):
    results = []
    for s in sizes:
        for t in timeouts:
            throughput = len(workload) / (1.0 + t * 0.1)
            latency = t * 10.0 + s * 0.5
            results.append({"max_batch_size": s, "batch_wait_timeout_s": t, "throughput": throughput, "latency": latency})
    return results
