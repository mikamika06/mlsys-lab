import numpy as np


def verify_distribution(target_logits, draft_logits, temperature=1.0):
    t_probs = np.exp(target_logits / temperature)
    t_probs /= np.sum(t_probs)
    d_probs = np.exp(draft_logits / temperature)
    d_probs /= np.sum(d_probs)
    return t_probs, d_probs
