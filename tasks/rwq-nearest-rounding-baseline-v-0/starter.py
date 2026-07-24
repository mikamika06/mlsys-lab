import numpy as np


def quantize_dequant_rtn_v0(W: np.ndarray, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-group asymmetric affine 4-bit round-to-nearest quantization — the
    V=0 (no learned rounding perturbation) baseline.

    Returns (codes, W_dq):
      codes -- uint8 array, same shape as W, values in [0, 15]
      W_dq  -- float64 array, same shape as W, dequantized reconstruction
    """
    raise NotImplementedError('your code here')
