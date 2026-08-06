def measure_acceptance_rate(draft_tokens, target_tokens):
    if not draft_tokens:
        return 0.0
    accepted = 0
    for d, t in zip(draft_tokens, target_tokens):
        if d == t:
            accepted += 1
        else:
            break
    return float(accepted / len(draft_tokens))


def analyze_domain_shift(in_domain_evals, ood_evals):
    in_rates = [measure_acceptance_rate(d, t) for d, t in in_domain_evals]
    ood_rates = [measure_acceptance_rate(d, t) for d, t in ood_evals]

    avg_in = float(sum(in_rates) / len(in_rates)) if in_rates else 0.0
    avg_ood = float(sum(ood_rates) / len(ood_rates)) if ood_rates else 0.0
    drop = float(avg_in - avg_ood)

    return {
        "avg_in_domain_acceptance": avg_in,
        "avg_ood_acceptance": avg_ood,
        "acceptance_drop": drop,
        "is_severe_collapse": bool(drop >= 0.30)
    }
