import numpy as np

def apply_frequency_presence_penalty(
    logits: np.ndarray,
    token_counts: np.ndarray,
    freq_penalty: float,
    presence_penalty: float
) -> np.ndarray:
    """
    Apply OpenAI's frequency + presence penalty to a vector of logits.

    Parameters
    ----------
    logits : np.ndarray
        Raw logits for each token, shape (vocab_size,).
    token_counts : np.ndarray
        Integer counts of how many times each token has appeared in the prompt,
        same shape as ``logits``.
    freq_penalty : float
        Penalty coefficient applied per occurrence of a token.
    presence_penalty : float
        Additional penalty applied if the token appears at least once.

    Returns
    -------
    np.ndarray
        Penalised logits, dtype float64 and same shape as ``logits``.
    """
    # Ensure correct dtypes
    logits = np.asarray(logits, dtype=np.float64)
    token_counts = np.asarray(token_counts, dtype=np.int64)

    # Presence mask: 1.0 if count > 0 else 0.0
    presence_mask = (token_counts > 0).astype(np.float64)

    # Total penalty per token
    penalty = token_counts * freq_penalty + presence_mask * presence_penalty

    return logits - penalty
