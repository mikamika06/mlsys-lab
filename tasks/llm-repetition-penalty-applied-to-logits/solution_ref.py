def apply_repetition_penalty(logits: list[float],
                             seen_tokens: list[int],
                             penalty: float) -> list[float]:
    """
    Apply the Hugging Face style repetition penalty to a vector of logits.

    Parameters
    ----------
    logits : list[float]
        List of logits.
    seen_tokens : list[int]
        Token indices that have already appeared in the context.
    penalty : float
        Penalty factor p > 1.0.

    Returns
    -------
    list[float]
        New logits list with the penalty applied only to tokens in `seen_tokens`.
    """
    out = list(logits)
    seen_set = set(seen_tokens)
    for i in range(len(out)):
        if i in seen_set:
            val = out[i]
            if val > 0:
                out[i] = val / penalty
            elif val <= 0:
                out[i] = val * penalty
    return out
