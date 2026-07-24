import numpy as np


def stable_variance(x: np.ndarray) -> float:
    # BUG: this is the textbook one-pass formula Var = E[x^2] - E[x]^2.
    # On data with a large offset, both terms are huge and nearly equal,
    # so their difference is dominated by float64 rounding noise (and can
    # even come out negative).
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    mean = np.sum(x) / n
    mean_sq = np.sum(x * x) / n
    return float(mean_sq - mean * mean)
