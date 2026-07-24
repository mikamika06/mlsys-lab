import numpy as np

def causal_alibi_logits(logits: np.ndarray, alibi_bias: np.ndarray) -> np.ndarray:
    """
    Compute attention probabilities with a causal mask and ALiBi bias.

    Parameters
    ----------
    logits : np.ndarray
        Raw attention logits of shape (seq_len, seq_len).
    alibi_bias : np.ndarray
        Linear bias to add to each logit, same shape as ``logits``.

    Returns
    -------
    probs : np.ndarray
        Row‑wise softmax probabilities after applying the causal mask and bias.
    """
    if logits.shape != alibi_bias.shape:
        raise ValueError("logits and alibi_bias must have identical shapes")
    seq_len = logits.shape[0]
    # Causal mask: -inf for future positions, 0 otherwise
    mask = np.triu(np.full_like(logits, fill_value=-np.inf), k=1)
    masked = logits + alibi_bias + mask
    # Row‑wise softmax with numerical stability
    max_vals = np.max(masked, axis=-1, keepdims=True)
    exp_shift = np.exp(masked - max_vals)
    probs = exp_shift / np.sum(exp_shift, axis=-1, keepdims=True)
    return probs.astype(np.float64)
