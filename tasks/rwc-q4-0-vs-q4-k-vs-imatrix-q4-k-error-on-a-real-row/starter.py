import numpy as np


def compare_q4_variants(x: np.ndarray, w: np.ndarray) -> tuple:
    """
    x: (32,) float64 real weight block.
    w: (32,) float64 positive importance weights.

    Returns (errors, best_idx):
      errors   - float64 array of shape (3,): importance-weighted MSE (using w)
                 of [Q4_0, Q4_K, Imatrix-Q4_K] reconstructions of x.
      best_idx - int index (0, 1, or 2) of the smallest entry in errors.
    """
    raise NotImplementedError('your code here')
