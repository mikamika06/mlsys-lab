from divcheck.divergence import compute_divergence_distribution


def evaluate_regression_gate(set_a, set_b, k, max_divergence_fraction):
    counts = compute_divergence_distribution(set_a, set_b)
    total_prompts = sum(counts)
    if total_prompts == 0:
        return {"passed": True, "early_divergence_fraction": 0.0, "early_count": 0, "total_prompts": 0}

    early_count = sum(counts[:k])
    fraction = early_count / float(total_prompts)
    passed = fraction <= max_divergence_fraction

    return {
        "passed": bool(passed),
        "early_divergence_fraction": float(fraction),
        "early_count": int(early_count),
        "total_prompts": int(total_prompts)
    }
