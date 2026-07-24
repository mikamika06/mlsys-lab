import numpy as np


def percentile_clip_best(x: np.ndarray, percentile_grid: np.ndarray, qmax: int):
    """
    Return (index, mse): the index into percentile_grid, and the resulting
    reconstruction MSE, of the clip percentile that minimizes MSE when
    clipping |x| at that percentile, symmetric-quantizing with qmax codes,
    and dequantizing. See task.md for the exact formulas.
    """
    raise NotImplementedError('your code here')
