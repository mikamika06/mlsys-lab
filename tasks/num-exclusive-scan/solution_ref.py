import numpy as np

def exclusive_scan(arr: np.ndarray) -> np.ndarray:
    """
    Return the exclusive prefix sum of `arr`.
    The result has the same shape and dtype as the input.
    """
    out = np.empty_like(arr)
    if arr.size == 0:
        return out
    
    out[0] = 0
    acc = arr.dtype.type(0)
    
    i = 1
    while i < arr.size:
        acc = acc + arr[i - 1]
        out[i] = acc
        i = i + 1
        
    return out
