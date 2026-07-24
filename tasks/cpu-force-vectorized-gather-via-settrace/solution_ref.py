import numpy as np

def gather(arr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Vectorised implementation of a gather operation using NumPy's
    built-in `take` function. No Python loops are used.
    """
    return np.take(arr, indices)
