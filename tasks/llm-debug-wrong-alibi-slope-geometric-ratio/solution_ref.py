def alibi_slopes(n: int) -> list[float]:
    slopes = []
    for h in range(n):
        slopes.append(2.0 ** (-8.0 * h / n))
    return slopes
