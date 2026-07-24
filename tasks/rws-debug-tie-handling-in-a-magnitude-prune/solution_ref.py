import numpy as np

def magnitude_prune_mask(weights: np.ndarray, keep_fraction: float) -> np.ndarray:
    """
    Return a boolean mask selecting the top‑`keep_fraction` fraction of weights by absolute value.
    Ties are broken stably according to the original index order.
    """
    n = len(weights)
    k = int(np.ceil(keep_fraction * n))
    if k <= 0:
        return np.zeros(n, dtype=bool)
    # stable sort indices by descending |weights|
    idx = np.argsort(-np.abs(weights), kind='mergesort')
    mask = np.zeros(n, dtype=bool)
    mask[idx[:k]] = True
    return mask
