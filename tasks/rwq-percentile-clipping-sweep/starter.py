import math

def percentile_clip_best(x: list[float], percentile_grid: list[float], qmax: int):
    """
    Return (index, mse): the index into percentile_grid, and the resulting
    reconstruction MSE, of the clip percentile that minimizes MSE when
    clipping |x| at that percentile, symmetric-quantizing with qmax codes,
    and dequantizing. See task.md for the exact formulas.
    """
    raise NotImplementedError('your code here')
