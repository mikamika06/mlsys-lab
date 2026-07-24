import numpy as np

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
    mask = np.zeros_like(logits, dtype=bool)
    for i, allowed in enumerate(allowed_sets):
        mask[i, list(allowed)] = True
    masked_logits = np.where(mask, logits, -np.inf)
    return np.argmax(masked_logits, axis=1)
