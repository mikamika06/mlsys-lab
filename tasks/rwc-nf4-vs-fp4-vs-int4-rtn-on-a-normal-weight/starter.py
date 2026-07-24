import numpy as np


def nf4_fp4_int4_best(w: np.ndarray):
    """
    w: 1-D array of weight values (assumed roughly normally distributed,
        but the function must work on whatever it's given).

    Reconstruct `w` three ways and compute each scheme's MSE:
      - NF4: absmax-scale w into [-1, 1], snap to the 16-level NF4
        codebook (quantiles of a standard normal), rescale back.
      - FP4: same absmax scaling, snapped to a 16-level FP4 codebook.
      - INT4 (RTN): a plain 4-bit AFFINE (asymmetric, min/max) uniform
        grid over 16 levels -- round-to-nearest quantization with no
        codebook shaping at all.

    Return (errors, best):
      errors: np.array([mse_nf4, mse_fp4, mse_int4]).
      best: the name of the scheme with the lowest MSE, one of
        "NF4", "FP4", "INT4".
    """
    raise NotImplementedError('your code here')
