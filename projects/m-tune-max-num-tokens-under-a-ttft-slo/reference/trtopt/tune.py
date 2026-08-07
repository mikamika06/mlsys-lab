import numpy as np
from trtopt.simulate import simulate_performance


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
