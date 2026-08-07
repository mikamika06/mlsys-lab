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
