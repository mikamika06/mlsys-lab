import numpy as np


def sparsegpt_24_prune(W: np.ndarray, X: np.ndarray, damp: float = 0.01):
    """SparseGPT-style 2:4 structured pruning: Hessian-diagonal mask
    selection + one-shot OBS/OBC compensation update.

    W: (O, I) float64 weight matrix, I % 4 == 0.
    X: (n, I) float64 calibration activations.
    damp: relative Hessian damping.

    1. H = X.T @ X; H += damp * mean(diag(H)) * identity.
    2. Hinv = inv(H); diag_hinv = diag(Hinv).
    3. Mask: for every row and every contiguous group of 4 columns,
       scores = W[row, group]**2 / diag_hinv[group]; prune (mask=0) the
       2 lowest-scoring columns, keep (mask=1) the other 2.
    4. Compensation: per row, let S = that row's pruned indices
       (mask[row] == 0). w_S = W[row, S].
       delta = -(Hinv[:, S] @ solve(Hinv[np.ix_(S, S)], w_S))
       W_hat[row] = W[row] + delta; W_hat[row, S] = 0.0 exactly.

    Returns (mask, W_hat): mask (O, I) with entries 0/1, W_hat (O, I)
    float64.
    """
    raise NotImplementedError('your code here')
