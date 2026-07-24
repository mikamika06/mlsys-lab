import numpy as np


def apply_wanda_mask(W: np.ndarray, M: np.ndarray, X: np.ndarray):
    """
    Apply a (precomputed) Wanda pruning mask M to weights W, and report
    both the pruned layer's output and how much it deviates from the
    unpruned layer's output on the same activations.

    Y = (W ⊙ M) @ X
    R = W @ X - Y     (the output residual introduced by pruning)

    W, M: (d_out, d_in). X: (d_in, n).
    Returns (Y, R), each (d_out, n).
    """
    W = np.asarray(W, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    Y = (W * M) @ X
    R = W @ X - Y
    return Y, R
