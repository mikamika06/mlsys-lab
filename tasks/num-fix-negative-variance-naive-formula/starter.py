import numpy as np


def stable_variance(x: np.ndarray) -> float:
    # TODO: replace unstable moment formula with a stable online algorithm
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x * x) - np.mean(x) ** 2)
