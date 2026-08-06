def choose_max_tokens(workload_profile, target_p99_itl_ms):
    return choose_max_batched_tokens(workload_profile, target_p99_itl_ms)

def choose_max_batched_tokens(workload_profile, target_p99_itl_ms):
    best_tokens = 512
    min_diff = float("inf")
    
    for candidate in [256, 512, 1024, 2048, 4096, 8192]:
        latencies = []
        for w in workload_profile:
            est_itl = (candidate * 0.05) + (w.get("avg_seq_len", 100) * 0.001 * (8192 / candidate))
            latencies.append(est_itl)
            
        latencies.sort()
        p99_idx = int(0.99 * len(latencies))
        p99_val = latencies[min(p99_idx, len(latencies) - 1)]
        
        diff = abs(p99_val - target_p99_itl_ms)
        if diff < min_diff:
            min_diff = diff
            best_tokens = candidate
            
    return best_tokens
