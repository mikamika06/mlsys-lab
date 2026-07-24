import numpy as np

def drop_step_prune(weights: np.ndarray, mask: np.ndarray, drop_frac: float) -> np.ndarray:
    """
    Return a new boolean mask after dropping the bottom fraction of live weights.
    
    Parameters
    ----------
    weights : np.ndarray
        1‑D array of weight values.
    mask : np.ndarray
        Boolean or integer mask of the same length; True/1 indicates live.
    drop_frac : float
        Fraction in [0,1] of live weights to prune.
    
    Returns
    -------
    np.ndarray
        New boolean mask with the specified fraction of smallest‑magnitude live
        weights removed.  The input mask is not modified.
    """
    # Ensure we work on a copy so inputs are untouched
    new_mask = mask.copy()
    if drop_frac <= 0:
        return new_mask.astype(bool)
    
    live_indices = np.nonzero(mask)[0]
    n_live = len(live_indices)
    if n_live == 0:
        return new_mask.astype(bool)

    k = int(np.floor(n_live * drop_frac))
    if k == 0:
        return new_mask.astype(bool)

    # Find the indices of the k smallest magnitudes among live weights
    live_abs = np.abs(weights[live_indices])
    sorted_idx = np.argsort(live_abs)
    indices_to_drop = live_indices[sorted_idx[:k]]

    new_mask[indices_to_drop] = False
    return new_mask.astype(bool)
