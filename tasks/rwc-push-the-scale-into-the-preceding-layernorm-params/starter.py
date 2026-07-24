import numpy as np

def fuse_scale_into_layernorm(ln_weight, ln_bias, ln_eps, scale,
                               linear_weight, linear_bias):
    """Fold per-feature scale into preceding LayerNorm gamma/beta and into
    the next Linear weight, so the explicit scale node can be removed.

    Returns (new_ln_weight, new_ln_bias, new_linear_weight).
    Do not modify input arrays in place.
    """
    raise NotImplementedError('Implement scale-to-LN and scale-to-Linear fusion')
