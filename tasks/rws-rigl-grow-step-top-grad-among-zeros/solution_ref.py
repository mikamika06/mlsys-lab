import numpy as np

def rigl_grow_step(gradients: np.ndarray,
                   mask: np.ndarray,
                   grow_count: int) -> np.ndarray:
    """
    Grow the top‑|grad| dormant connections.

    Parameters
    ----------
    gradients : np.ndarray, shape (n,)
        Gradient vector for all parameters.
    mask : np.ndarray, shape (n,), bool or integer
        Current binary mask of active weights.
    grow_count : int
        Number of dormant connections to activate.

    Returns
    -------
    new_mask : np.ndarray, same shape and dtype as `mask`
        Updated mask with exactly `grow_count` additional ones at the indices
        having the largest absolute gradients among those that were zero.
    """
    gradients = np.asarray(gradients)
    mask = np.asarray(mask, dtype=bool)
    if gradients.shape != mask.shape:
        raise ValueError("gradients and mask must have the same shape")
    zero_pos = np.where(~mask)[0]
    if len(zero_pos) == 0 or grow_count <= 0:
        return mask.copy()
    abs_grad = np.abs(gradients[zero_pos])
    if grow_count > len(zero_pos):
        grow_count = len(zero_pos)
    # Select top‑grow_count indices by absolute gradient magnitude
    top_idx_within_zero = np.argpartition(-abs_grad, grow_count-1)[:grow_count]
    top_indices = zero_pos[top_idx_within_zero]
    new_mask = mask.copy()
    new_mask[top_indices] = True
    return new_mask
