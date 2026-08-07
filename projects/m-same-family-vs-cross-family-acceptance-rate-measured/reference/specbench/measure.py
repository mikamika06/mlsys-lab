import numpy as np


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
            if prob_ratio >= 1.0 or np.random.rand() < prob_ratio:
                accepted += 1
            else:
                break
    return float(accepted) / float(total)
