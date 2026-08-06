import numpy as np


def stable_variance(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    n = len(arr)
    total = 0.0
    for i in range(n):
        total += float(arr[i])
    mean = total / n
    dev_sum = 0.0
    for i in range(n):
        dev = float(arr[i]) - mean
        dev_sum += dev * dev
    return float(dev_sum / n)
