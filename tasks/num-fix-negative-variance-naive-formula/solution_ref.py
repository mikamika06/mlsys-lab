import numpy as np


def stable_variance(x: np.ndarray) -> float:
    values = np.asarray(x, dtype=np.float64)
    if values.size == 0:
        return float("nan")

    centered = values - values[0]

    mean = 0.0
    m2 = 0.0
    count = 0

    for value in centered:
        count += 1
        delta = value - mean
        mean += delta / count
        m2 += delta * (value - mean)

    return float(m2 / count)
