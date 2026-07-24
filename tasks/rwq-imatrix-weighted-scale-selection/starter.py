import numpy as np


def imatrix_best_scale(x: np.ndarray, w: np.ndarray, scale_grid: np.ndarray, qmin: int, qmax: int) -> int:
    """
    Return the index into scale_grid minimizing the imatrix-weighted
    quantization error sum(w * (x - xhat)**2), where xhat is x symmetric-
    quantized at that scale and clipped to [qmin, qmax]. See task.md.
    """
    raise NotImplementedError('your code here')
