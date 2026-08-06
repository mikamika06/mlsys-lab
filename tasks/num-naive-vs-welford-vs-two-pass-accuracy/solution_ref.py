def welford_variance(data: list[float]) -> float:
    """Population variance via Welford's single-pass online algorithm."""
    n = 0
    mean = 0.0
    m2 = 0.0
    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2
    return m2 / n
