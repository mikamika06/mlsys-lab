import numpy as np


def stable_variance(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    dev = x - mean
    return float(np.mean(dev * dev))
