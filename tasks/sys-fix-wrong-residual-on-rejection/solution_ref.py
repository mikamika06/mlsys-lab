def residual_distribution(p: list[float], q: list[float]) -> list[float]:
    """
    The speculative-decoding rejection (residual) distribution: elementwise
    max(p - q, 0), renormalized to sum to 1.
    """
    diffs = []
    total = 0.0
    for pi, qi in zip(p, q):
        d = pi - qi
        if d < 0.0:
            d = 0.0
        diffs.append(d)
        total += d

    if total == 0.0:
        return [0.0 for _ in p]

    return [d / total for d in diffs]
