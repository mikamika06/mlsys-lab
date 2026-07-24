import numpy as np


def awq_clip_search(W: np.ndarray, group_size: int, clip_ratios: np.ndarray, bits: int = 4):
    """AutoAWQ-style post-scaling clip-ratio search.

    W: shape (rows, cols), cols an exact multiple of group_size.
    clip_ratios: 1-D array of candidate ratios in (0, 1], tried in order.
    bits: quantization bit width (symmetric, qmax = 2**(bits-1) - 1).

    For each group of group_size columns and each candidate ratio r:
      clipped_amax = amax(group) * r
      scale = clipped_amax / qmax
      clip the group to [-clipped_amax, clipped_amax], quantize/dequantize
      with `scale`, and measure MSE against the ORIGINAL (unclipped) group.

    Returns (best_idx, best_mse):
      best_idx: int64 array, shape (rows, cols // group_size), index into
        clip_ratios of the ratio minimizing that group's MSE.
      best_mse: float64 array, same shape, the MSE achieved at best_idx.
    """
    raise NotImplementedError('your code here')
