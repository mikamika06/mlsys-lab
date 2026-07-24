import numpy as np


def wanda_score_mask(W: np.ndarray, X: np.ndarray, sparsity: float) -> np.ndarray:
    """Wanda importance score and per-output-row pruning mask.

    W: shape (d_out, d_in).
    X: shape (n_samples, d_in), calibration activations.
    sparsity: fraction of each row's columns to prune (keep the top
      1 - sparsity fraction, ranked by score).

    Score: S_ij = |W_ij| * ||X[:, j]||_2  (L2 norm of column j over samples).
    Per output row i, keep the k = round((1 - sparsity) * d_in) highest-scoring
    columns (at least 1); mask out the rest.

    Returns a boolean array, same shape as W, True where the weight is kept.
    """
    raise NotImplementedError('your code here')
