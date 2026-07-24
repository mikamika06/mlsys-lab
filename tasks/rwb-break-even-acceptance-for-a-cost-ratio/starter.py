import numpy as np


def break_even_alpha(configs: np.ndarray) -> np.ndarray:
    """
    configs: (n, 2) array, each row [c, k] -- draft-to-target cost ratio
    c (0 <= c < 1) and draft block length k (non-negative integer, as a
    float).

    For each row, solve for the unique alpha* in [0, 1] with
    sum_{i=0}^{k} alpha*^i == 1 + k*c (numerically -- there is no closed
    form for general k).

    Returns an (n,) float64 array of alpha* values.
    """
    raise NotImplementedError('your code here')
