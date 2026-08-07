import numpy as np


def compute_acceptance_rate(tokens, draft_logits, target_logits, gamma=4):
    accepted_total = 0
    total_proposed = 0
    n = len(tokens)
    idx = 0
    while idx < n - gamma:
        k = min(gamma, n - idx)
        for j in range(k):
            total_proposed += 1
            p_draft = np.exp(draft_logits[idx + j] - np.max(draft_logits[idx + j]))
            p_draft /= np.sum(p_draft)
            p_target = np.exp(target_logits[idx + j] - np.max(target_logits[idx + j]))
            p_target /= np.sum(p_target)
            token = tokens[idx + j]
            ratio = p_target[token] / (p_draft[token] + 1e-12)
            if ratio >= 1.0 or ratio >= 0.5:
                accepted_total += 1
            else:
                break
        idx += max(1, k)
    if total_proposed == 0:
        return 0.0
    return float(accepted_total / total_proposed)
