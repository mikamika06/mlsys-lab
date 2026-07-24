import numpy as np

def can_reshape_view(shape, strides, newshape):
    """
    Return True iff reshaping an array with the given shape and strides
    yields a view of the original data.
    """
    itemsize = np.int64().itemsize
    max_offset = sum((s - 1) * st for s, st in zip(shape, strides))
    total_bytes = max_offset + itemsize
    n_elements = (total_bytes + itemsize - 1) // itemsize
    base = np.arange(n_elements, dtype=np.int64)
    try:
        arr = np.lib.stride_tricks.as_strided(base, shape=shape, strides=strides)
        reshaped = arr.reshape(newshape)
        return np.may_share_memory(arr, reshaped)
    except Exception:
        return False
