import numpy as np


def qint4_granularity_mse(W: np.ndarray, group_size: int = 32):
    """Compare reconstruction error of symmetric int4 quantization (levels
    -7..7, scale = amax / 7) at two granularities.

    W: shape (rows, cols), cols an exact multiple of group_size.

    Returns (mse_per_axis, mse_per_group):
      mse_per_axis: MSE using one scale per row (per output-channel / per-axis).
      mse_per_group: MSE using one scale per group_size-column group.
    """
    raise NotImplementedError('your code here')
