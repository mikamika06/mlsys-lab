import numpy as np
import math

def masked_greedy(logits: np.ndarray,
                  allowed_sets) -> np.ndarray:
    """
    Return the greedy token indices after masking disallowed logits.

    Parameters
    ----------
    logits : np.ndarray, shape (n_steps, vocab_size)
        Logit scores for each token at each decoding step.
    allowed_sets : Iterable[Iterable[int]]
        For each step a collection of token indices that are permitted.

    Returns
    -------
    np.ndarray, shape (n_steps,)
        The index of the chosen token for each step.
    """
    n_steps, vocab_size = logits.shape
    result = np.zeros(n_steps, dtype=logits.dtype)
    for i, allowed in enumerate(allowed_sets):
        allowed_set = set(allowed)
        best_val = -math.inf
        best_idx = 0
        first = True
        for j in range(vocab_size):
            if j in allowed_set:
                val = logits[i, j]
                if first or val > best_val:
                    best_val = val
                    best_idx = j
                    first = False
        result[i] = best_idx
    return result
