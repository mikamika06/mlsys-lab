def compute_throughput(latencies, num_tokens):
    if not latencies:
        return 0.0
    total_time = max(latencies)
    return float(num_tokens) / float(total_time)


def compute_throughput_ratio(baseline_latencies, candidate_latencies, num_tokens, concurrency):
    t_base = compute_throughput(baseline_latencies, num_tokens)
    t_cand = compute_throughput(candidate_latencies, num_tokens)
    if t_base == 0.0:
        return 0.0
    return float(t_cand) / float(t_base)
