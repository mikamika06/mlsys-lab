def stable_variance(x: list[float]) -> float:
    if not x:
        return float("nan")

    centered = [val - x[0] for val in x]

    mean = 0.0
    m2 = 0.0
    count = 0

    for value in centered:
        count += 1
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)

    return float(m2 / count)
