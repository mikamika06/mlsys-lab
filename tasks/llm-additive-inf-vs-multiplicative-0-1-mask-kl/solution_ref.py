import numpy as np

def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities with additive -inf masking.

    Parameters
    ----------
    logits : np.ndarray
        Raw attention scores of shape (batch, seq_len).
    mask : np.ndarray
        Boolean or integer mask of the same shape. Positions where mask==0 are masked out.

    Returns
    -------
    probs : np.ndarray
        Softmax probabilities with masked entries set to zero and each row summing to one.
    """
    mask_bool = mask.astype(bool)
    # Additive -inf masking
    masked_logits = np.where(mask_bool, logits, -np.inf)

    # Numerical stability: subtract per‑row max before exp
    max_per_row = np.max(masked_logits, axis=-1, keepdims=True)
    exp_vals = np.exp(masked_logits - max_per_row)

    sum_exp = np.sum(exp_vals, axis=-1, keepdims=True)
    probs = exp_vals / sum_exp
    return probs
