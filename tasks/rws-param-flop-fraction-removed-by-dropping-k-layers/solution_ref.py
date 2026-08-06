def removed_and_remaining(param_counts, k):
    """
    Compute the fraction of parameters removed and the remaining ratio
    when dropping the first `k` layers from a model.

    Parameters
    ----------
    param_counts : np.ndarray
        1‑D array of non‑negative integers giving the number of parameters per layer.
    k : int
        Number of layers to drop from the beginning (0 ≤ k ≤ len(param_counts)).

    Returns
    -------
    tuple[float, float]
        (removed_fraction, remaining_ratio)
    """
    import numpy as np

    total = 0
    n = len(param_counts)
    for i in range(n):
        total += param_counts[i]

    if k <= 0:
        removed = 0
    else:
        k_clamped = k if k < n else n
        removed = 0
        for i in range(k_clamped):
            removed += param_counts[i]

    return (removed / total, (total - removed) / total)
