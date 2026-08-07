import numpy as np


def simulate_performance(max_num_tokens, batching_type, arrival_rate, prefill_lengths):
    tokens = np.array(prefill_lengths)
    if batching_type == "static":
        ttft = np.mean(tokens) * (1.0 + 0.05 * (max_num_tokens / 2048.0))
        throughput = arrival_rate * 0.8
    else:
        penalty = max(0.0, (max_num_tokens - 1024) / 4096.0)
        ttft = np.mean(tokens) * (0.5 + 0.1 * penalty)
        throughput = arrival_rate * min(1.0, max_num_tokens / 512.0)
    return float(ttft), float(throughput)


def tune_max_tokens(candidate_tokens, slo_ttft, batching_type, arrival_rate, prefill_lengths):
    ttfts = []
    throughputs = []
    for m in candidate_tokens:
        t, tp = simulate_performance(m, batching_type, arrival_rate, prefill_lengths)
        ttfts.append(t)
        throughputs.append(tp)
    ttfts = np.array(ttfts)
    throughputs = np.array(throughputs)
    valid = ttfts <= slo_ttft
    if not np.any(valid):
        return int(candidate_tokens[np.argmin(ttfts)])
    valid_indices = np.where(valid)[0]
    best_idx = valid_indices[np.argmax(throughputs[valid_indices])]
    return int(candidate_tokens[best_idx])
