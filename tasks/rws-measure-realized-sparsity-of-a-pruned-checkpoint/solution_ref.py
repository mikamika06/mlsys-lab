import numpy as np

def count_zeros_and_sparsity(W: np.ndarray) -> tuple[int, float]:
    """
    Return the number of zero elements in W and its realized sparsity ratio.
    """
    zeros = int(np.count_nonzero(W == 0))
    ratio = zeros / W.size if W.size else 0.0
    return zeros, float(ratio)
