def calculate_zero_preemption_max_seqs(workload_profile, total_blocks):
    qps = workload_profile["qps"]
    avg_tokens = workload_profile["avg_tokens"]
    block_size = workload_profile["block_size"]
    blocks_per_req = (avg_tokens + block_size - 1) // block_size
    max_safe_seqs = total_blocks // max(1, blocks_per_req)
    optimal_seqs = int(min(max_safe_seqs, max(1, int(qps * 2.0))))
    return optimal_seqs
