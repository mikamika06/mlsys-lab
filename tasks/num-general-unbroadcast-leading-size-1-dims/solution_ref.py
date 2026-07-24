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
    grad = np.asarray(grad, dtype=np.float64)

    # 1) sum out extra leading dims that `shape` does not have
    n_extra = grad.ndim - len(shape)
    for _ in range(n_extra):
        grad = grad.sum(axis=0)

    # 2) sum (with keepdims) every remaining axis where shape[i] == 1 but
    #    grad grew wider there
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad.reshape(shape)
