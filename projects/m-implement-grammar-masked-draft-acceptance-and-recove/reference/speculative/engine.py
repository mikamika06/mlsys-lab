import numpy as np


def accept_draft(draft_tokens, target_probs, grammar_masks):
    accepted = []
    for i, token in enumerate(draft_tokens):
        mask = grammar_masks[i] if i < len(grammar_masks) else None
        if mask is not None and token not in mask:
            break
        prob = target_probs[i][token] if i < len(target_probs) else 0.0
        if prob <= 0.0:
            break
        accepted.append(token)
    return accepted
