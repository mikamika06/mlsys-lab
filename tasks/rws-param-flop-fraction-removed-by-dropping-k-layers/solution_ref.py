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

    total = param_counts.sum()
    if k <= 0:
        removed = 0
    else:
        # Clamp k to the number of layers to avoid out‑of‑bounds.
        k_clamped = min(k, len(param_counts))
        removed = param_counts[:k_clamped].sum()
    return (removed / total, (total - removed) / total)
