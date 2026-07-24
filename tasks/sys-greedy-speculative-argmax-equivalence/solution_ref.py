import numpy as np

def greedy_speculative(draft_logits: np.ndarray,
                       target_logits: np.ndarray) -> list[int]:
    """
    Return the token indices chosen by speculative decoding.
    draft_logits and target_logits are 2‑D arrays of shape (T, V).
    """
    draft_top = np.argmax(draft_logits, axis=1)
    target_top = np.argmax(target_logits, axis=1)
    return list(np.where(draft_top == target_top, draft_top, target_top))
