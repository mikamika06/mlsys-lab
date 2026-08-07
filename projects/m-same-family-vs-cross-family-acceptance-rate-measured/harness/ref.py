import numpy as np

PAIRS = [
    {
        "draft": [101, 202, 303, 404],
        "target": [101, 202, 303, 404],
        "probs": [{"target_prob": 0.9, "draft_prob": 0.8}, {"target_prob": 0.8, "draft_prob": 0.8}, {"target_prob": 0.85, "draft_prob": 0.7}, {"target_prob": 0.95, "draft_prob": 0.9}],
        "is_same_family": True,
    },
    {
        "draft": [101, 505, 606, 707],
        "target": [101, 202, 303, 404],
        "probs": [{"target_prob": 0.9, "draft_prob": 0.2}, {"target_prob": 0.1, "draft_prob": 0.6}, {"target_prob": 0.05, "draft_prob": 0.8}, {"target_prob": 0.2, "draft_prob": 0.7}],
        "is_same_family": False,
    },
    {
        "draft": [101, 202, 808, 909],
        "target": [101, 202, 303, 404],
        "probs": [{"target_prob": 0.9, "draft_prob": 0.8}, {"target_prob": 0.8, "draft_prob": 0.8}, {"target_prob": 0.15, "draft_prob": 0.5}, {"target_prob": 0.1, "draft_prob": 0.4}],
        "is_same_family": False,
    },
]

def compute_acceptance_rate(draft_tokens, target_tokens, probabilities):
    if not draft_tokens or not target_tokens:
        return 0.0
    accepted = 0
    total = len(draft_tokens)
    for d_tok, t_tok, p in zip(draft_tokens, target_tokens, probabilities):
        if d_tok == t_tok:
            accepted += 1
        else:
            prob_ratio = float(p.get("target_prob", 0.0)) / max(float(p.get("draft_prob", 1e-5)), 1e-5)
            if prob_ratio >= 1.0 or 0.5 < prob_ratio:
                accepted += 1
            else:
                break
    return float(accepted) / float(total)

def family_gap_ratio(same_family_rates, cross_family_rates):
    mean_same = float(np.mean(same_family_rates)) if same_family_rates else 0.0
    mean_cross = float(np.mean(cross_family_rates)) if cross_family_rates else 0.0
    if mean_cross == 0.0:
        return float("inf") if mean_same > 0 else 1.0
    return mean_same / mean_cross
