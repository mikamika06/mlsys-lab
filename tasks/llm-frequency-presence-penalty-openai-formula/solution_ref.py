def apply_frequency_presence_penalty(
    logits: list[float],
    token_counts: list[int],
    freq_penalty: float,
    presence_penalty: float
) -> list[float]:
    """
    Apply OpenAI's frequency + presence penalty to a vector of logits.

    Parameters
    ----------
    logits : list[float]
        Raw logits for each token.
    token_counts : list[int]
        Integer counts of how many times each token has appeared in the prompt,
        same length as ``logits``.
    freq_penalty : float
        Penalty coefficient applied per occurrence of a token.
    presence_penalty : float
        Additional penalty applied if the token appears at least once.

    Returns
    -------
    list[float]
        Penalised logits, same length as ``logits``.
    """
    penalised = []
    for l, c in zip(logits, token_counts):
        presence = 1.0 if c > 0 else 0.0
        penalty = c * freq_penalty + presence * presence_penalty
        penalised.append(l - penalty)
    return penalised
