import numpy as np
from numpy.lib.stride_tricks import as_strided


def sliding_window(x: np.ndarray, w: int) -> np.ndarray:
    """Return a zero-copy (N-w+1, w) view of all length-w windows of 1D `x`.

    Must share memory with `x`, keep its dtype, work for non-contiguous inputs,
    and raise ValueError when w < 1 or w > len(x).
    """
    raise NotImplementedError('your code here')
