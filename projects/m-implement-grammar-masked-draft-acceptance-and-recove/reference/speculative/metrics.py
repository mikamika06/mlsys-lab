import numpy as np
from speculative.acceptance import compute_accepted_length


def measure_acceptance_loss(unmasked_drafts, target_probs_list, grammar_masks):
    accepted_lengths = []
    total_draft_tokens = 0
    for draft_tokens, target_probs, grammar_mask in zip(unmasked_drafts, target_probs_list, grammar_masks):
        k = len(draft_tokens)
        total_draft_tokens += k
        draft_probs = [np.ones((len(tp),)) / len(tp) for tp in target_probs]
        acc = compute_accepted_length(draft_tokens, target_probs, draft_probs, grammar_mask)
        accepted_lengths.append(acc)
    mean_accepted = sum(accepted_lengths) / max(1, len(accepted_lengths))
    loss = 1.0 - (mean_accepted / max(1, total_draft_tokens / max(1, len(unmasked_drafts))))
    return {"mean_accepted": float(mean_accepted), "acceptance_loss": float(loss)}
