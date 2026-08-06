import numpy as np


def validate_nm_sparsity(weight, n, m, dim=-1):
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
    valid = np.all(non_zeros <= n)
    return bool(valid)
