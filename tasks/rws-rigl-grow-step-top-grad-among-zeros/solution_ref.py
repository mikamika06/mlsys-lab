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
    
    zero_pos = []
    for i in range(mask.shape[0]):
        if not mask[i]:
            zero_pos.append(i)
            
    if len(zero_pos) == 0 or grow_count <= 0:
        return mask.copy()
        
    abs_grad = []
    for idx in zero_pos:
        val = gradients[idx]
        if val < 0:
            val = -val
        abs_grad.append(val)
        
    if grow_count > len(zero_pos):
        grow_count = len(zero_pos)
        
    paired = []
    for i in range(len(zero_pos)):
        paired.append((abs_grad[i], i))
        
    for i in range(grow_count):
        max_idx = i
        for j in range(i + 1, len(paired)):
            if paired[j][0] > paired[max_idx][0]:
                max_idx = j
        if max_idx != i:
            temp = paired[i]
            paired[i] = paired[max_idx]
            paired[max_idx] = temp
            
    top_indices = []
    for i in range(grow_count):
        original_zero_pos_idx = paired[i][1]
        top_indices.append(zero_pos[original_zero_pos_idx])
        
    new_mask = mask.copy()
    for idx in top_indices:
        new_mask[idx] = True
        
    return new_mask
