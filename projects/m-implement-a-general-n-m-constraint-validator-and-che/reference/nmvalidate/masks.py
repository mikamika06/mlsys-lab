import numpy as np


def extract_nm_mask(weight, n, m, dim=-1):
    w = np.asarray(weight)
    if m <= 0 or n < 0 or n > m:
        raise ValueError("Invalid N:M parameters")
    if dim < 0:
        dim += w.ndim
    shape = w.shape
    block_size = shape[dim]
    if block_size % m != 0:
        raise ValueError("Dimension size must be divisible by M")
    axes = list(range(w.ndim))
    axes.append(axes.pop(dim))
    w_perm = np.transpose(w, axes)
    new_shape = w_perm.shape[:-1] + (-1, m)
    reshaped = w_perm.reshape(new_shape)
    non_zeros = np.sum(reshaped != 0, axis=-1)
    over_limit = non_zeros > n
    if np.any(over_limit):
        raise ValueError("Weight matrix violates N:M sparsity constraint")
    mask_reshaped = (reshaped != 0).astype(np.uint8)
    mask_perm = mask_reshaped.reshape(w_perm.shape)
    inv_axes = [0] * w.ndim
    for i, ax in enumerate(axes):
        inv_axes[ax] = i
    mask = np.transpose(mask_perm, inv_axes)
    return mask
