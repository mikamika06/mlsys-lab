import math
import numpy as np

def pyramidkv_allocation(total_budget: int, num_layers: int) -> np.ndarray:
    """
    Compute a pyramidal KV budget allocation.

    Parameters
    ----------
    total_budget : int
        Total number of KV slots to distribute.
    num_layers : int
        Number of transformer layers.

    Returns
    -------
    numpy.ndarray
        1‑D array of length `num_layers` with integer allocations that sum to
        `total_budget`.  Lower indices receive at least as many slots as higher
        ones.
    """
    if total_budget < 0 or num_layers <= 0:
        raise ValueError("Both arguments must be positive integers")

    S = 0
    for i in range(1, num_layers + 1):
        S += i

    base_list = []
    base_sum = 0
    for i in range(1, num_layers + 1):
        val = (total_budget * i) // S
        base_list.append(val)
        base_sum += val

    remainder = total_budget - base_sum

    for i in range(remainder):
        base_list[num_layers - 1 - i] += 1

    return np.array(base_list, dtype=np.int64)
