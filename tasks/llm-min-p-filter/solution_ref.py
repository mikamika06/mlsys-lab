def minp_filter(probs: list[float], min_p: float) -> list[bool]:
    """
    Return a boolean mask indicating which tokens have probability at least
    `min_p` times the maximum probability in `probs`.
    """
    if not probs:
        return []

    max_prob = probs[0]
    for x in probs:
        if x > max_prob:
            max_prob = x
    threshold = min_p * max_prob

    out = []
    for x in probs:
        out.append(x >= threshold)
    return out
