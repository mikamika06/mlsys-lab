import numpy as np


def compute_accepted_length(draft_tokens, target_probs, draft_probs, grammar_mask):
    k = len(draft_tokens)
    accepted = 0
    for i in range(k):
        token = draft_tokens[i]
        if i < len(grammar_mask) and not grammar_mask[i][token]:
            break
        p_target = target_probs[i][token]
        p_draft = max(draft_probs[i][token], 1e-6)
        ratio = p_target / p_draft
        if ratio >= 1.0:
            accepted += 1
        else:
            r = np.random.default_rng(42 + i).random()
            if r < ratio:
                accepted += 1
            else:
                break
    return accepted
