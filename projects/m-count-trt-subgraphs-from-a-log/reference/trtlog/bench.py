def compare_eps(cuda_latencies, trt_latencies):
    """Compare CUDA EP and TRT EP latencies head-to-head."""
    c_mean = sum(cuda_latencies) / len(cuda_latencies)
    t_mean = sum(trt_latencies) / len(trt_latencies)
    return {"cuda_mean": c_mean, "trt_mean": t_mean, "latency_ratio": c_mean / t_mean}

def simulate_cache_sessions(session_logs):
    """Simulate multiple sessions and check timing cache reuse."""
    results = []
    for log in session_logs:
        from trtlog.parser import parse_log
        parsed = parse_log(log)
        results.append(parsed["cache_reused"])
    return results
