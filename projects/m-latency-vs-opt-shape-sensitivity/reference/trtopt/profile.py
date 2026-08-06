"""Optimization profile splitting and evaluation."""

import numpy as np

def split_wide_profile(wide_profile):
    """Split wide profile into low and high profiles."""
    min_s = list(wide_profile["min"])
    opt_s = list(wide_profile["opt"])
    max_s = list(wide_profile["max"])

    split_dim = 0
    min_b = min_s[split_dim]
    max_b = max_s[split_dim]

    mid_b = (min_b + max_b) // 2
    opt_low_b = max(min_b, (min_b + mid_b) // 2)
    opt_high_b = min(max_b, (mid_b + max_b) // 2)

    low_profile = {
        "min": [min_b] + min_s[1:],
        "opt": [opt_low_b] + opt_s[1:],
        "max": [mid_b] + max_s[1:],
    }
    high_profile = {
        "min": [mid_b] + min_s[1:],
        "opt": [opt_high_b] + opt_s[1:],
        "max": [max_b] + max_s[1:],
    }

    return [low_profile, high_profile]

def evaluate_profile_latency(profiles, query_shapes, cost_fn):
    """Evaluate latency over query shapes using optimal profile selection."""
    total_latency = 0.0
    for shape in query_shapes:
        shape_arr = np.array(shape, dtype=np.float64)
        best_cost = float("inf")
        for prof in profiles:
            p_min = np.array(prof["min"], dtype=np.float64)
            p_max = np.array(prof["max"], dtype=np.float64)
            if np.all(shape_arr >= p_min) and np.all(shape_arr <= p_max):
                p_opt = np.array(prof["opt"], dtype=np.float64)
                c = cost_fn(shape_arr, p_opt)
                if c < best_cost:
                    best_cost = c
        if best_cost == float("inf"):
            p_opt = np.array(profiles[0]["opt"], dtype=np.float64)
            best_cost = cost_fn(shape_arr, p_opt)
        total_latency += best_cost
    return float(total_latency)
