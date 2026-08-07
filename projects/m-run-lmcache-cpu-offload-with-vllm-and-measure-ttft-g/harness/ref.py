import numpy as np

REQUESTS_SET = [
    [{"prompt_len": 512, "cache_hit_rate": 0.5}, {"prompt_len": 1024, "cache_hit_rate": 0.9}],
    [{"prompt_len": 2048, "cache_hit_rate": 0.2}, {"prompt_len": 256, "cache_hit_rate": 0.0}],
    [{"prompt_len": 4096, "cache_hit_rate": 0.75}, {"prompt_len": 512, "cache_hit_rate": 0.5}]
]

CONFIGS_SET = [
    {"chunk_size": 128},
    {"chunk_size": 256},
    {"chunk_size": 512}
]

LOG_BATCHES = [
    ["t1,1.5,10000", "t2,60.2,2000"],
    ["t3,0.5,50000", "t4,2.0,15000"],
    ["t5,80.0,500", "t6,0.1,80000"]
]

TOKEN_LENGTH_SETS = [
    [512, 1024, 2048],
    [256, 512, 1024],
    [1024, 2048, 4096]
]

BANDWIDTHS = [1000.0, 2000.0, 500.0]

def measure_ttft_gain(requests, config):
    ratios = []
    for req in requests:
        prompt_len = req["prompt_len"]
        hit_rate = req["cache_hit_rate"]
        chunk_size = config.get("chunk_size", 256)
        base_time = prompt_len * 0.05
        transfer_overhead = ((prompt_len * hit_rate) / chunk_size) * 0.002 + 0.005 if hit_rate > 0 else 0
        offload_time = (prompt_len * (1 - hit_rate) * 0.05) + transfer_overhead
        ratio = offload_time / base_time if base_time > 0 else 1.0
        ratios.append(float(ratio))
    return {"latency_ratios": ratios, "mean_ratio": float(np.mean(ratios))}

def diagnose_transfer_log(log_lines):
    results = []
    for line in log_lines:
        parts = line.strip().split(",")
        if len(parts) < 3:
            results.append("unknown")
            continue
        duration = float(parts[1])
        bytes_transferred = float(parts[2])
        throughput = bytes_transferred / (duration + 1e-6)
        if throughput < 10.0:
            results.append("pcie_bottleneck")
        elif duration > 50.0:
            results.append("lock_contention")
        else:
            results.append("healthy")
    return results

def find_optimal_chunk_size(token_lengths, bandwidth_profile):
    best_chunk = 64
    min_cost = float("inf")
    candidates = [64, 128, 256, 512, 1024, 2048]
    for c in candidates:
        cost = 0.0
        for length in token_lengths:
            chunks = np.ceil(length / c)
            overhead = chunks * 0.001 + (c / bandwidth_profile)
            cost += overhead
        if cost < min_cost:
            min_cost = cost
            best_chunk = c
    return int(best_chunk)
