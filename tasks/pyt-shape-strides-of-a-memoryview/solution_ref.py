import numpy as np

def memoryview_info(arr):
    """Return shape, strides, itemsize, ndim, format of arr's memoryview."""
    mv = memoryview(arr)
    return {
        "shape": mv.shape,
        "strides": mv.strides,
        "itemsize": mv.itemsize,
        "ndim": mv.ndim,
        "format": mv.format,
    }
