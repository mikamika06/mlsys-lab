import numpy as np

def exclusive_scan(arr: np.ndarray) -> np.ndarray:
    """
    Return the exclusive prefix sum of `arr`.
    The result has the same shape and dtype as the input.
    """
    if arr.size == 0:
        return np.empty_like(arr)
    out = np.empty_like(arr)
    out[0] = 0
    if arr.size > 1:
        # Cumulative sum of all but the last element, shifted right by one.
        out[1:] = np.cumsum(arr[:-1])
    return out
