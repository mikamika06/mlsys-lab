import numpy as np


def all_reduce_mean_grads(grads: list[np.ndarray]) -> np.ndarray:
    """
    All-reduce a list of per-worker gradients into the mean gradient,
    as every worker would see it after a sum-then-divide all-reduce.
    """
    total = np.zeros_like(grads[0], dtype=np.float64)
    for g in grads:
        total += g
    n = len(grads)
    return total / n
