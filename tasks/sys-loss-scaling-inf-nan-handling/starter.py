import numpy as np


def unscale_and_check(scaled_grads, scale):
    """Unscale gradients from loss scaling and decide whether to skip the step.

    scaled_grads: list of float32 ndarrays, may contain inf/nan.
    scale: python float loss-scale factor (a power of two).

    Returns (skip, unscaled_grads):
      skip: bool, True iff any element in any array is inf or nan.
      unscaled_grads: list of float32 ndarrays, each = scaled_grads[i] / scale,
      computed unconditionally regardless of skip.
    """
    raise NotImplementedError('your code here')
