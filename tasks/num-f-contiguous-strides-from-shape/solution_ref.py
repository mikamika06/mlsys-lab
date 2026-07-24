import numpy as np

def f_contiguous_strides_from_shape(shape):
    """
    Return the Fortran‑contiguous strides (in bytes) for an array of the given shape.
    """
    itemsize = np.dtype(np.float64).itemsize
    if len(shape) == 0:
        return ()
    strides = [itemsize]
    for i in range(1, len(shape)):
        strides.append(strides[-1] * shape[i-1])
    return tuple(strides)
