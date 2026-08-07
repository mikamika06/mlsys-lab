def compute_accepted_length(draft_tokens, target_probs, draft_probs, grammar_masks, random_samples):
    accepted_length = 0
    for i, token in enumerate(draft_tokens):
        if not grammar_masks[i][token]:
            break
        p = target_probs[i][token]
        q = draft_probs[i][token]
        if p >= q or random_samples[i] < (p / q):
            accepted_length += 1
        else:
            break
    return accepted_length
