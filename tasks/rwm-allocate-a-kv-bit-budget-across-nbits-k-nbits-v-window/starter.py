import numpy as np


def choose_kv_budget(
    K: np.ndarray,
    V: np.ndarray,
    candidates: list,
    byte_budget: int,
    group_size: int,
) -> int:
    """
    candidates is a list of (nbits_K, nbits_V, R) triples. Return the index
    of the feasible (byte cost <= byte_budget) config with the smallest
    total reconstruction MSE, as defined in task.md.
    """
    raise NotImplementedError('your code here')
