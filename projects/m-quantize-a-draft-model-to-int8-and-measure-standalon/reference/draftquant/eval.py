import numpy as np


def simulate_acceptance(target_logits, draft_logits, int8_draft_logits):
    orig_probs = np.exp(draft_logits) / np.sum(np.exp(draft_logits), axis=-1, keepdims=True)
    int8_probs = np.exp(int8_draft_logits) / np.sum(np.exp(int8_draft_logits), axis=-1, keepdims=True)
    target_probs = np.exp(target_logits) / np.sum(np.exp(target_logits), axis=-1, keepdims=True)

    orig_match = np.sum(np.argmax(orig_probs, axis=-1) == np.argmax(target_probs, axis=-1)) / len(target_logits)
    int8_match = np.sum(np.argmax(int8_probs, axis=-1) == np.argmax(target_probs, axis=-1)) / len(target_logits)
    return {"orig_acceptance": float(orig_match), "int8_acceptance": float(int8_match)}
