CONFIGS = [
    {"requests": 100, "tokens_per_req": 128, "base_latency": 12.0},
    {"requests": 200, "tokens_per_req": 64, "base_latency": 8.5},
    {"requests": 150, "tokens_per_req": 256, "base_latency": 20.0},
]

def run_benchmark(tp_degree, workload_config):
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

def compute_scaling_efficiency(tp1_result, tp2_result):
    t1 = tp1_result["throughput"]
    t2 = tp2_result["throughput"]
    ratio = t2 / max(t1, 0.0001)
    efficiency = ratio / 2.0
    return {
        "throughput_ratio": float(ratio),
        "scaling_efficiency": float(efficiency)
    }
