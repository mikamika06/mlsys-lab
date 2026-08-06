import numpy as np


def unbroadcast(grad, shape):
    """Reduce `grad` (which has the shape produced by broadcasting a tensor
    of shape `shape` up to grad.shape) back down to `shape`, by summing over
    every axis that broadcasting introduced or stretched.

    Two independent kinds of axes have to be summed away:
      1. Extra LEADING axes that `shape` doesn't have at all (numpy inserts
         these on the left when aligning ranks).
      2. Axes where `shape` itself is 1 but `grad` is wider (numpy stretched
         a size-1 axis).

    Returns a float64 array with exactly `len(shape)` dimensions, equal to
    `shape`.
    """
    grad_arr = np.asarray(grad, dtype=np.float64)
    out = np.zeros(shape, dtype=np.float64)

    g_shape = grad_arr.shape
    g_size = grad_arr.size
    g_rank = len(g_shape)

    g_strides = []
    acc = 1
    for dim in reversed(g_shape):
        g_strides.append(acc)
        acc *= dim
    g_strides.reverse()

    target_shape = (1,) * (g_rank - len(shape)) + tuple(shape)

    target_strides = []
    acc = 1
    for dim in reversed(target_shape):
        target_strides.append(acc)
        acc *= dim
    target_strides.reverse()

    flat_grad = grad_arr.ravel()
    flat_out = out.ravel()

    for idx in range(g_size):
        rem = idx
        out_idx = 0
        for r in range(g_rank):
            dim_size = g_shape[r]
            coord = rem // g_strides[r]
            rem %= g_strides[r]
            if target_shape[r] != 1:
                out_idx += coord * target_strides[r]

        flat_out[out_idx] += flat_grad[idx]

    return out
