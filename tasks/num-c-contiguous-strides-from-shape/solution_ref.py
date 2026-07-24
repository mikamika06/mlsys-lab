import numpy as np

def c_contig_strides(shape, dtype):
    arr = np.empty(shape, dtype=dtype)
    return tuple(arr.strides)
