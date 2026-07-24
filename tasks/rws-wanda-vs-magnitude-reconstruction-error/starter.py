import numpy as np


def wanda_vs_magnitude_error(W: np.ndarray, X: np.ndarray, sparsity: float):
    """
    Compute (e_wanda, e_magnitude): the squared Frobenius output error
    ||W@X - (W*M)@X||_F^2 caused by pruning W at the given per-row
    sparsity, once with the Wanda mask (|W_ij| * ||X row j||_2) and once
    with the pure-magnitude mask (|W_ij|), as described in task.md.
    """
    raise NotImplementedError('your code here')
