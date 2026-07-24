import numpy as np

def apply_repetition_penalty(logits: np.ndarray,
                             seen_tokens: list[int],
                             penalty: float) -> np.ndarray:
    """
    Apply the Hugging Face style repetition penalty to a vector of logits.

    Parameters
    ----------
    logits : np.ndarray
        1‑D array of shape (V,) with dtype float64.
    seen_tokens : list[int]
        Token indices that have already appeared in the context.
    penalty : float
        Penalty factor p > 1.0.

    Returns
    -------
    np.ndarray
        New logits array with the penalty applied only to tokens in `seen_tokens`.
    """
    logits = np.asarray(logits, dtype=np.float64)
    out = logits.copy()
    if len(seen_tokens) == 0:
        return out

    mask = np.zeros_like(out, dtype=bool)
    mask[list(seen_tokens)] = True
    pos_mask = mask & (out > 0)
    neg_mask = mask & (out <= 0)

    out[pos_mask] /= penalty
    out[neg_mask] *= penalty
    return out
