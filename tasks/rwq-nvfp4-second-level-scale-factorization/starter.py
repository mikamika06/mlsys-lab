import numpy as np


def nvfp4_block_scales(W: np.ndarray, group_size: int, per_tensor_scale: float) -> np.ndarray:
    """NVFP4 second-level block-scale factorization.

    W: 1-D float64 array, len(W) a multiple of group_size.
    group_size: block size (NVFP4 uses 16).
    per_tensor_scale: positive float, the tensor's single FP32 scale.

    For each contiguous block of `group_size` elements:
      1. raw_scale = max(|block|) / (6.0 * per_tensor_scale)   (6.0 is
         E2M1's largest representable magnitude)
      2. round raw_scale to the nearest representable non-negative
         E4M3 magnitude (1 sign, 4 exponent, 3 mantissa; bias 7; no
         infinities; NaN only at exponent==15, mantissa==7).

    Returns the array of per-block E4M3-quantized scales, shape
    (len(W) // group_size,), float64.
    """
    raise NotImplementedError('your code here')
