def unbroadcast(grad, shape):
    """Reduce `grad` (which has the shape produced by broadcasting a tensor
    of shape `shape` up to grad.shape) back down to `shape`, by summing over
    every axis that broadcasting introduced or stretched. Returns a float64
    array with exactly `len(shape)` dimensions, equal to `shape`."""
    raise NotImplementedError('your code here')
