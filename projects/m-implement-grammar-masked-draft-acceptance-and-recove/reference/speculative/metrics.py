import numpy as np

def measure_acceptance_loss(batch_draft_tokens, batch_target_probs, batch_draft_probs, batch_grammar_masks, batch_random_samples):
    from speculative.acceptance import compute_accepted_length
    total_loss = 0.0
    n = len(batch_draft_tokens)
    if n == 0:
        return 0.0

    for i in range(n):
        len_masked = compute_accepted_length(
            batch_draft_tokens[i],
            batch_target_probs[i],
            batch_draft_probs[i],
            batch_grammar_masks[i],
            batch_random_samples[i]
        )
        unmasked = [np.ones_like(m, dtype=bool) for m in batch_grammar_masks[i]]
        len_unmasked = compute_accepted_length(
            batch_draft_tokens[i],
            batch_target_probs[i],
            batch_draft_probs[i],
            unmasked,
            batch_random_samples[i]
        )
        total_loss += (len_unmasked - len_masked)

    return float(total_loss / n)
