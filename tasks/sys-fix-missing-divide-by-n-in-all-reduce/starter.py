import numpy as np


def all_reduce_mean_grads(grads: list[np.ndarray]) -> np.ndarray:
    """
    BUG: this all-reduce sums every worker's gradient but never divides
    by the world size N, so it returns the SUM instead of the MEAN.
    Fix it so it returns the correct mean gradient.
    """
    total = np.zeros_like(grads[0], dtype=np.float64)
    for g in grads:
        total += g
    return total  # <-- missing divide by len(grads)
