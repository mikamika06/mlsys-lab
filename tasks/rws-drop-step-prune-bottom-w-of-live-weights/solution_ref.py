import math
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
    new_mask = mask.copy()
    if drop_frac <= 0:
        return new_mask.astype(bool)
    
    live_indices = []
    for i in range(len(mask)):
        if mask[i]:
            live_indices.append(i)
            
    n_live = len(live_indices)
    if n_live == 0:
        return new_mask.astype(bool)

    k = int(math.floor(n_live * drop_frac))
    if k == 0:
        return new_mask.astype(bool)

    live_abs = []
    for idx in live_indices:
        val = weights[idx]
        if val < 0:
            live_abs.append(-val)
        else:
            live_abs.append(val)

    sorted_idx = sorted(range(n_live), key=lambda i: live_abs[i])
    indices_to_drop = [live_indices[sorted_idx[i]] for i in range(k)]

    new_mask[indices_to_drop] = False
    return new_mask.astype(bool)
