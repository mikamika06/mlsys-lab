import math


def compute_acceptance_metrics(matches, entropy):
    total_positions = 100
    matched_count = len(matches)
    acceptance_rate = float(matched_count) / float(max(1, total_positions))
    entropy_val = float(entropy)
    gap = max(0.0, 1.0 - (entropy_val / 5.0))
    return {
        "acceptance_rate": acceptance_rate,
        "entropy_gap": gap,
        "matched_count": float(matched_count)
    }


def classify_workload(acceptance_rate, entropy_gap):
    if acceptance_rate > 0.4 and entropy_gap > 0.5:
        return "copy-heavy"
    return "creative-writing"
