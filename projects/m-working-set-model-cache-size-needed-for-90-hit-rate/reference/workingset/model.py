import numpy as np

def compute_working_set_size(traces, target_hit_rate=0.9):
    unique_tokens = sorted(list(set(traces)))
    if not unique_tokens:
        return 0
    counts = {}
    for t in traces:
        counts[t] = counts.get(t, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    total_requests = len(traces)
    cumulative = 0
    active_set = 0
    for token, freq in sorted_items:
        cumulative += freq
        active_set += 1
        if cumulative / total_requests >= target_hit_rate:
            return active_set
    return len(unique_tokens)
