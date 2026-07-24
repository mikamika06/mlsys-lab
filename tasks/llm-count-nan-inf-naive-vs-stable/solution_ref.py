import numpy as np

def count_nonfinite_in_naive_softmax(x: np.ndarray) -> int:
    """
    Compute the naïve softmax of each row in `x` and count how many entries
    are not finite (NaN or Inf).
    """
    exp_x = np.exp(x, dtype=np.float64)
    sum_exp = np.sum(exp_x, axis=1, keepdims=True)
    softmax = exp_x / sum_exp
    nonfinite_mask = ~np.isfinite(softmax)
    return int(np.count_nonzero(nonfinite_mask))
