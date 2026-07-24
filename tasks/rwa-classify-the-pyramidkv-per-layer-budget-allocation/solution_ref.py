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

    weights = np.arange(1, num_layers + 1)          # bottom layer gets largest weight
    S = int(weights.sum())
    base = (total_budget * weights) // S           # floor division gives integer array
    remainder = total_budget - int(base.sum())

    # Distribute remaining slots starting from the bottom layer
    for i in range(remainder):
        base[num_layers - 1 - i] += 1

    return base.astype(np.int64)
