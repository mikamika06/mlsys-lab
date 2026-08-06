from offload.simulator import estimate_load_time, estimate_recompute_time


def select_optimal_tier(config, hw, tiers, prefix_len, max_latency_s):
    """Select the storage tier or recompute option minimizing latency within budget."""
    candidates = {
        "recompute": estimate_recompute_time(config, hw, prefix_len)
    }
    for name, tier in tiers.items():
        candidates[name] = estimate_load_time(config, tier, prefix_len)

    valid = {k: v for k, v in candidates.items() if v <= max_latency_s}
    if valid:
        best_name = min(valid, key=lambda k: (valid[k], k))
    else:
        best_name = min(candidates, key=lambda k: (candidates[k], k))
    return best_name


def evaluate_workload_latencies(config, hw, tiers, workload_prefix_lengths):
    """Evaluate latencies across all available tiers for a sequence of workload prefix lengths."""
    result = {
        "recompute": [estimate_recompute_time(config, hw, L) for L in workload_prefix_lengths]
    }
    for name, tier in tiers.items():
        result[name] = [estimate_load_time(config, tier, L) for L in workload_prefix_lengths]
    return result
